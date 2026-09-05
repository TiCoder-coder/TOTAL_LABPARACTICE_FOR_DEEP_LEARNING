# PHASE 53 — ATTENTION HEATMAPS

## Kế hoạch render, kiểm định và chuẩn hóa heatmap cho temporal self-attention của ba Final Transformer seeds từ raw artifacts `ATTENTION_EXTRACTION-v1`

**Project:** UCI Appliances Energy Prediction  
**Task:** Multivariate Time-Series Regression  
**Model:** Attention-Aware Transformer Encoder for regression  
**Forecasting contract:** Sequence-to-One, One-Step-Ahead  
**Raw attention source:** `ATTENTION_EXTRACTION-v1`  
**Raw dense tensor contract:** `[case, layer, head, query, source]`  
**Per-map matrix contract:** `[query, source] = [L,L]`  
**Heatmap x-axis:** Source/Key historical position  
**Heatmap y-axis:** Query historical position  
**Phase ID:** `PHASE_53_ATTENTION_HEATMAPS`  
**Output version:** `ATTENTION_HEATMAPS-v1`  
**Phase trước:** `Phase_52_Attention_extraction.md`  
**Phase sau:** `Phase_54_Last-query_attention.md`

---

# 1. Vai trò của Phase 53

Phase53 biến các raw dense attention matrices đã được kiểm định ở Phase52 thành các **scientifically traceable heatmaps** có:

```text
đúng query/source orientation
đúng temporal order
đúng layer/head identity
đúng target/case identity
đúng lag mapping
đúng scale semantics
đúng provenance
```

Mục tiêu:

```text
1. Verify raw Phase52 dense-attention artifacts/checksums.
2. Freeze rendering/axis/color-scale contract trước khi render.
3. Render per-head heatmaps cho toàn bộ frozen dense-case set.
4. Giữ nguyên attention values; không smoothing/thresholding/interpolation.
5. Tạo deterministic grids giúp xem all heads/layers cho từng case/seed.
6. Tạo deterministic cross-seed grids cho report-selected shared worst cases.
7. Chuẩn hóa source/query axis bằng timestamp/lag semantics.
8. Tách “absolute probability view” và “case-shared comparison view”.
9. Tạo catalog/index để mọi hình truy ngược được raw tensor.
10. Cho phép qualitative visual observations rất hạn chế, nhưng không rank head.
11. Không chọn case/head/seed vì heatmap đẹp.
12. Không re-extract attention.
13. Không dùng attention như raw-feature importance hay causal explanation.
14. Handoff visualization context sang Phase54, trong khi Phase54 vẫn dùng raw Phase52 last-query source.
```

Nguyên tắc:

\[
\boxed{
Frozen\ Raw\ Attention
+
Correct\ Orientation
+
Transparent\ Color\ Scaling
+
Per\text{-}Head\ Visualization
+
Deterministic\ Case\ Catalog
+
No\ Re\text{-}Extraction
+
No\ Head\ Selection
+
No\ Causal\ Claim
}
\]

---

# 2. Phase53 không tạo attention mới

Hard:

```text
model checkpoint loading for extraction = forbidden
new attention extraction               = forbidden
new Test prediction                    = forbidden.
```

Phase53 phải đọc trực tiếp:

```text
dense_case_attention_seed42.npz
dense_case_attention_seed123.npz
dense_case_attention_seed2026.npz
```

từ Phase52.

Nếu raw source thiếu/corrupt:

```text
STOP
→ resolve Phase52 artifact integrity.
```

Không rerun model trong Phase53.

---

# 3. Phase53 không thay thế raw source bằng image

Heatmap PNG/SVG chỉ là:

```text
derived visualization artifact.
```

Scientific source vẫn là:

```text
float32 raw attention arrays
+
raw checksums
+
case/order maps.
```

Không bao giờ:

```text
đọc pixel PNG
→ suy ngược attention value
```

cho downstream quantitative analysis.

---

# 4. Upstream hard gate

Required:

```text
phase_52_signoff.json
phase53_attention_heatmaps_handoff.json
raw_attention_checksums.json
attention_dense_case_order.csv
attention_relative_position_map.csv
attention_case_position_map.csv
attention_case_metadata.csv
```

Hard:

```text
phase53_ready = true.
```

Phase52:

```text
PASS
or
PASS_WITH_WARNING.
```

---

# 5. Required raw attention artifacts

Exactly:

```text
dense_case_attention_seed42.npz
dense_case_attention_seed123.npz
dense_case_attention_seed2026.npz.
```

Canonical shape per seed:

```text
[K_ATTN_CASES,N_layers,N_heads,L,L].
```

No head-averaged raw source.

---

# 6. Raw source verification

Before rendering:

```text
verify file SHA256
verify case-order SHA256
verify relative-position-map SHA256
verify final lock SHA256
verify shape
verify dtype=float32
verify same K across seeds
verify same case IDs/order across seeds
verify same N_layers/N_heads/L.
```

Any mismatch:

```text
STOP.
```

---

# 7. Minimal probability recheck

Phase52 already performed full attention probability verification.

Phase53 should perform a lightweight source-integrity recheck:

```text
finite
min >= numerical tolerance
max <= 1 + tolerance
sampled/full row sums ≈ 1
```

Preferred:

```text
full row-sum recheck
```

if computationally cheap for dense case set.

No renormalization.

---

# 8. Canonical matrix semantics

For one case/layer/head:

```text
M[q,s]
=
attention weight from query position q
to source/key position s.
```

Shape:

```text
[L,L].
```

---

# 9. Heatmap orientation is locked

Canonical visualization:

```text
x-axis = source/key position
y-axis = query position.
```

Therefore the raw matrix is plotted as:

```text
M
```

without transpose.

Forbidden:

```text
M.T
```

unless explicitly producing a separately labeled diagnostic, which Phase53 does not require.

---

# 10. Why orientation matters

If transposed:

```text
last-query row
```

becomes:

```text
last-source column
```

and interpretation becomes incorrect.

Phase53 must include an orientation unit test before official rendering.

---

# 11. Canonical temporal order

Both axes:

```text
position 0   = oldest historical input
position L-1 = newest historical input.
```

Display policy:

```text
source: oldest → newest from left to right
query:  oldest → newest from top to bottom.
```

Thus:

```text
top-left
=
oldest query attending to oldest source

bottom-right
=
newest query attending to newest source.
```

---

# 12. Image origin policy

Canonical renderer must place:

```text
row 0 at top
row L-1 at bottom.
```

This keeps newest query at the bottom.

Renderer config must explicitly record this orientation instead of relying on plotting-library defaults.

---

# 13. Heatmap x-axis meaning

Primary label:

```text
Source / key position in historical input
(oldest → newest).
```

Secondary tick labels should use:

```text
lag to forecast target
```

where possible.

---

# 14. Heatmap y-axis meaning

Primary label:

```text
Query position in historical input
(oldest → newest).
```

Query positions also map to their lag from the forecast target.

---

# 15. Relative lag mapping

Use frozen Phase52 map:

\[
LagSteps_p
=
H+(L-1-p).
\]

Project:

```text
H=1.
```

Therefore:

\[
LagSteps_p=L-p.
\]

No independently recomputed alternative mapping.

---

# 16. Lag-axis tick policy

Predeclare requested lag ticks:

```text
1
6
36
72
144
```

steps, retaining only those satisfying:

```text
1 <= lag <= L.
```

Interpretations:

```text
1   = 10 min
6   = 1 h
36  = 6 h
72  = 12 h
144 = 24 h.
```

---

# 17. Always include oldest position if not already represented

If final `L` is not one of the registered lag values, include:

```text
lag=L
```

as the oldest-input tick.

In current project candidate lookbacks are normally:

```text
36
72
144
```

but runtime final L is authoritative.

---

# 18. Tick position formula

For a lag `k` under H1:

\[
position=L-k.
\]

Use the frozen relative-position map rather than relying only on formula.

---

# 19. Tick density

Do not label every 10-minute position.

Primary heatmaps should have:

```text
approximately 3–6 meaningful ticks per axis.
```

This preserves readability.

---

# 20. Case-specific timestamp labels

Do not place all timestamps on heatmap axes.

Case-specific timestamp mapping remains available in:

```text
attention_case_position_map.csv.
```

For report heatmaps, title/caption contains:

```text
target timestamp
input start
input end.
```

---

# 21. No forecast target token on axes

The forecast target is not part of:

```text
[L,L]
```

attention matrix.

Do not add an extra row/column labeled target.

---

# 22. No causal-mask triangle shading

Final model uses:

```text
no causal mask.
```

Therefore Phase53 must not gray out or hide the upper triangle.

Upper-triangle values are valid attention among already-observed historical inputs.

---

# 23. Diagonal meaning

The main diagonal:

```text
q=s
```

represents same-position self-attention.

It may be visually prominent or not.

No assumption that strong diagonal is “better”.

---

# 24. Canonical heatmap does not overlay a diagonal line

Primary raw heatmaps should not obscure values.

Optional annotated report versions may add a very thin diagonal guide only if:

```text
annotation=true
```

is recorded.

Default:

```text
no overlay.
```

---

# 25. Last-query row highlighting

Because Phase54 specifically analyzes:

```text
q=L-1
```

Phase53 canonical heatmap does not need to emphasize it.

Optional report annotation may mark:

```text
bottom row = newest query
```

without altering raw colors.

No last-query quantitative interpretation in Phase53.

---

# 26. Color semantics

Attention weights are nonnegative probabilities.

Use a:

```text
sequential perceptually uniform color scale.
```

Do not use a diverging scale centered at zero.

---

# 27. No arbitrary aesthetic colormap switching by case

One project-wide colormap specification should be frozen in:

```text
attention_heatmap_render_config.json.
```

No case-specific colormap.

---

# 28. Colorbar label

Canonical:

```text
Attention weight
```

not:

```text
Importance
Causal contribution
Feature importance.
```

---

# 29. Color-scale problem

Attention values may be much smaller than 1 for diffuse heads.

A single:

```text
0..1
```

scale guarantees absolute comparability but may visually compress structure.

Per-map auto-scaling improves local visibility but destroys cross-map magnitude comparability.

Phase53 therefore defines two explicit render modes.

---

# 30. Render Mode A — FIXED_PROBABILITY

Canonical absolute view:

```text
vmin = 0
vmax = 1.
```

Name:

```text
FIXED_PROBABILITY.
```

Properties:

```text
absolute probability scale
cross-case comparable
cross-seed comparable
cross-layer comparable
cross-head comparable.
```

Limitation:

```text
diffuse low-magnitude structure may appear dark.
```

---

# 31. Mode A is the global comparability reference

At least all **report-selected shared top cases** must have a Mode A representation.

It is not mandatory to duplicate every appendix contact sheet in Mode A if storage/report volume becomes excessive, but the underlying raw values remain available.

---

# 32. Render Mode B — CASE_SHARED_SCALE

Primary structural comparison view for one case.

For a frozen target ID `c`:

\[
vmax_c
=
\max_{
seed,layer,head,q,s
}
A_{c}.
\]

Then every panel for that case across:

```text
all 3 seeds
all layers
all heads
```

uses:

```text
vmin=0
vmax=vmax_c.
```

---

# 33. Why CASE_SHARED_SCALE is preferred for case comparison

It improves visibility while preserving valid comparison **within the same target case** across:

```text
seeds
layers
heads.
```

It avoids deceptive per-panel autoscaling.

---

# 34. Mode B limitation

Color intensity cannot be directly compared across different cases if their:

```text
vmax_case
```

differs.

Every Mode B caption/title must state:

```text
Case-shared scale
vmax = <runtime value>.
```

---

# 35. Per-map auto-scaling is forbidden as a canonical output

Do not use:

```text
vmax = current head map max
```

independently for each panel in the official comparison grid.

Reason:

```text
two heads with very different absolute concentrations
could look equally bright.
```

---

# 36. Percentile clipping is not a canonical mode

Do not use:

```text
p95
p99
```

color clipping as the standard scientific heatmap.

It may hide high-attention cells.

No percentile-based scale is required in Phase53.

---

# 37. Log-scale attention heatmaps are not required

Avoid by default.

Raw probability scale and case-shared linear scale are sufficient.

---

# 38. No color normalization of raw values

The renderer passes raw:

```text
M[q,s]
```

to the color mapping.

No:

```text
row z-score
min-max normalization
softmax
renormalization.
```

---

# 39. No interpolation

Canonical:

```text
interpolation = none / nearest-neighbor equivalent.
```

No bicubic/bilinear smoothing.

Each displayed cell corresponds to one actual matrix cell.

---

# 40. No resampling the attention matrix

Do not resize the numerical matrix before plotting.

The visual canvas can resize pixels, but the underlying array remains:

```text
L×L.
```

---

# 41. Raster/vector output

Primary:

```text
PNG high-resolution.
```

Optional report export:

```text
SVG/PDF
```

if plotting stack supports it reliably.

Raw NPZ remains source of truth.

---

# 42. Image metadata

Every heatmap image should be traceable through a render manifest rather than relying only on file naming.

Required metadata:

```text
target ID
seed
layer
head
render mode
raw file SHA256
case row index
vmin/vmax
axis orientation
position-map SHA256.
```

---

# 43. Head/layer display IDs

Storage:

```text
layer_idx0=0..N-1
head_idx0=0..H-1.
```

Human display:

```text
Layer 1..N
Head 1..H.
```

Never mix zero-based/one-based IDs in filenames without explicit naming.

---

# 44. Canonical file naming

For individual map:

```text
CASE_<case_row_idx>_SEED_<seed>_LAYER_<display_layer>_HEAD_<display_head>_<mode>.png
```

Do not depend on long/raw target IDs as the only identifier.

Target ID is stored in manifest.

---

# 45. Case row index is stable

Use:

```text
attention_dense_case_order.csv
```

as authoritative case index.

No re-sorting by:

```text
attention pattern
heatmap brightness.
```

---

# 46. All dense cases must be rendered

Every frozen dense case from Phase52 should receive required scientific visualization coverage.

No omission because:

```text
map is flat
map is visually redundant
case is not interesting.
```

---

# 47. Visualization tiers

Phase53 uses three output tiers:

```text
V1 — Complete per-case/per-seed head-layer grids
V2 — Report-ready cross-seed comparison grids for deterministic shared top cases
V3 — Optional individual per-head maps for detailed appendix/debugging.
```

---

# 48. V1 — Per-case/per-seed grid

For each:

```text
case
seed
```

create one contact sheet:

```text
rows    = layers
columns = heads.
```

Each panel is:

```text
[L,L] query×source.
```

Use:

```text
CASE_SHARED_SCALE.
```

This yields:

```text
K_ATTN_CASES × 3
```

primary grid images.

---

# 49. Why rows=layers, columns=heads

This preserves the natural architecture hierarchy:

```text
vertical → depth
horizontal → head identity.
```

It also prevents generating thousands of standalone images as the only catalog.

---

# 50. V1 panel title

Compact:

```text
L1-H1
L1-H2
...
```

The grid-level title contains:

```text
target ID/display
target timestamp
seed
selection roles
regime labels
case-shared vmax.
```

---

# 51. V1 shared colorbar

One shared colorbar per grid.

Because all panels use the same:

```text
vmin/vmax.
```

Do not use one colorbar per panel in the main contact sheet.

---

# 52. V2 — Cross-seed comparison grids

For report-selected cases only.

Deterministic case set:

```text
Phase51 W2 shared ranks 1–5.
```

No heatmap-driven selection.

---

# 53. V2 layout

For each:

```text
case
layer
```

create grid:

```text
rows    = seeds [42,123,2026]
columns = heads [1..H].
```

Use:

```text
same CASE_SHARED_SCALE
```

for the case.

---

# 54. Why V2 is useful

It makes it easy to visually compare:

```text
same architectural head index
across seeds
```

while retaining the Phase52 caveat:

```text
same numeric head index
does not guarantee same semantic role.
```

The grid supports visual inspection only; Phase57 handles formal stability/head matching.

---

# 55. V2 report title

Include:

```text
Shared worst-error rank
target timestamp
layer
regime tags
scale mode.
```

Do not include:

```text
“best head”
“important head”.
```

---

# 56. V3 — Individual per-head files

Optional but recommended for:

```text
shared top5 report cases
```

and any case later needed in publication figures.

Each individual map uses:

```text
both Mode A and/or Mode B
```

as specified in render manifest.

---

# 57. V3 is not the only scientific source

Standalone images are convenience artifacts.

Quantitative work still uses raw NPZ.

---

# 58. FIXED_PROBABILITY report appendix

For the shared top5 cases, create:

```text
cross-seed/layer/head Mode A heatmaps
```

or one compact appendix grid per case.

This guarantees an absolute-probability reference.

---

# 59. No seed-averaged heatmap in Phase53

Because independently trained heads may be permuted functionally, do not average:

```text
seed42 Head1
seed123 Head1
seed2026 Head1
```

into one heatmap.

Phase57 will decide how to handle cross-seed head alignment.

---

# 60. No layer-averaged heatmap as a primary result

Averaging layers can erase depth-specific structure.

Not required.

---

# 61. Head-mean heatmaps

Optional derived overview:

```text
mean across heads within one layer and one seed
```

may be generated only if clearly labeled:

```text
DERIVED_HEAD_MEAN_OVERVIEW.
```

It must never replace per-head maps.

Preferred Phase53 core:

```text
per-head grids only.
```

---

# 62. No quantitative head ranking in Phase53

Do not rank heads by:

```text
peak intensity
entropy
recent mass
visual sharpness.
```

That belongs Phase55.

---

# 63. No quantitative seed-stability conclusion in Phase53

Visual similarity may be noted cautiously, but formal cross-seed stability belongs Phase57.

---

# 64. No error-conditioned attention statistics in Phase53

Cases already carry error/regime metadata for context.

Do not compute:

```text
high-error vs low-error attention differences.
```

That belongs Phase56.

---

# 65. No last-query quantitative analysis in Phase53

The bottom row is visible in heatmaps.

Do not perform:

```text
expected lag
recent mass
top source lag
```

analysis here.

Phase54 uses Phase52 raw last-query vectors.

---

# 66. Heatmap titles may include error context

Allowed metadata:

```text
actual y
seed-specific prediction
residual sign
shared-hardness rank
frozen regime labels.
```

Keep title concise; detailed error values can be in caption/catalog.

---

# 67. Do not use error values to alter rendering scale

Color scale depends only on:

```text
attention values
and
predeclared render mode.
```

Not:

```text
error magnitude.
```

---

# 68. Orientation unit test

Before official rendering, create a synthetic asymmetric matrix such as:

```text
M[1,3] = large marker
M[3,0] = second marker
```

with known query/source meanings.

Verify renderer places:

```text
M[1,3]
at
query row 1, source column 3
```

and not transposed.

---

# 69. Orientation test must not depend only on visual inspection

The render helper should return/record:

```text
plotted_array_shape
transpose_applied=false
x_semantics=SOURCE
y_semantics=QUERY
origin policy.
```

A unit test asserts configuration.

---

# 70. Real-source orientation audit

For a deterministic raw map:

```text
read M[q,s]
```

for several fixed coordinates and verify render metadata maps them to:

```text
x=s
y=q.
```

No image-pixel OCR or reverse reading required.

---

# 71. Last-query-row orientation audit

For a deterministic dense case:

```text
M[L-1,:]
```

must correspond to the visually bottom row under canonical origin.

Record:

```text
last_query_display_edge=BOTTOM.
```

---

# 72. Diagonal orientation audit

The self-attention diagonal:

```text
q=s
```

must run:

```text
top-left → bottom-right.
```

This is a strong transpose/origin sanity check.

---

# 73. Color-scale audit

For every image:

```text
record mode
vmin
vmax
raw_map_min
raw_map_max
scale_scope.
```

Hard:

```text
raw_map_max <= vmax
```

for canonical non-clipped views.

---

# 74. Mode A scale audit

Expected:

```text
vmin=0
vmax=1.
```

Exactly.

---

# 75. Mode B scale audit

For case `c`:

```text
vmax
=
global max across all dense maps
for that target across 3 seeds/layers/heads.
```

All Mode B images/grids for that case must use identical `vmax`.

---

# 76. Case-shared vmax must be computed before individual panel rendering

This prevents per-panel accidental auto-scale.

---

# 77. Colorbar audit

Every grid/image must include:

```text
Attention weight
```

and scale bounds discoverable from metadata.

---

# 78. Axis audit

Required labels:

```text
x = Source / key historical position
y = Query historical position.
```

Report versions may additionally label lag ticks.

---

# 79. Lag tick audit

For every tick:

```text
displayed lag
↔
position index
```

must match:

```text
attention_relative_position_map.csv.
```

No manually typed approximate labels.

---

# 80. Case title provenance

Case title values should come from:

```text
attention_case_metadata.csv
```

not ad hoc joins.

---

# 81. Regime title provenance

Regime labels are frozen Phase50 labels already carried through Phase51/52.

No recomputation.

---

# 82. Seed-specific error title provenance

If shown, use frozen Phase49/51 error metadata.

No recalculation needed except integrity checks.

---

# 83. Heatmap catalog

Create one canonical:

```text
attention_heatmap_catalog.csv
```

with one row per generated image.

Fields:

```text
image_id
path
case_row_idx
target_id
target_timestamp
seed
layer_scope
head_scope
render_mode
vmin
vmax
raw_attention_file
raw_attention_sha256
case_order_sha256
position_map_sha256
grid_layout
report_selected
status.
```

---

# 84. Case visualization index

Create:

```text
attention_heatmap_case_index.csv
```

one row per dense case:

```text
case row
target ID
selection roles
shared rank
regime labels
seed42 grid path
seed123 grid path
seed2026 grid path
cross-seed grid paths if report selected
fixed-probability appendix path if report selected.
```

---

# 85. Render config artifact

Create:

```text
attention_heatmap_render_config.json
```

before rendering.

Must freeze:

```text
matrix orientation
origin
axis semantics
tick policy
color scale type
Mode A bounds
Mode B formula
interpolation
aspect policy
figure dimensions
resolution
font sizing policy
colorbar label
grid layouts
report-selected case rule.
```

---

# 86. Render config fingerprint

Create:

```text
attention_heatmap_render_config_fingerprint.json.
```

A render-config change after official images exist requires:

```text
new render version
```

not silent overwrite.

---

# 87. Figure dimensions

Use consistent dimensions based on:

```text
N_layers
N_heads
L
```

and grid type.

Do not use different figure sizes based on whether a case “looks interesting”.

---

# 88. Aspect ratio

Individual attention matrix should be displayed as a square matrix where practical:

```text
equal query/source spatial scale.
```

Contact sheet cell proportions remain consistent.

---

# 89. Resolution

Primary PNG should be high enough for:

```text
report insertion
zoomed inspection.
```

Recommended:

```text
300 DPI equivalent
```

for report figures.

Exact pixel dimensions are an engineering setting recorded in render config.

---

# 90. No anti-aliased interpolation of matrix cells

Text/vector elements can be anti-aliased.

Matrix cell interpolation remains:

```text
none/nearest.
```

---

# 91. Case-shared scale manifest

Create:

```text
attention_heatmap_case_scale_manifest.csv
```

with:

```text
case_row_idx
target_id
case_attention_max
case_attention_min
vmin_mode_B=0
vmax_mode_B
seed42_max
seed123_max
seed2026_max
status.
```

No interpretation.

---

# 92. Report-selected case set

Canonical:

```text
Phase51 W2 shared ranks 1–5.
```

If fewer than 5 shared cases exist:

```text
use all available
```

and record count.

No replacement from other ranks.

---

# 93. Report-selected case manifest

Create:

```text
attention_heatmap_report_cases.csv
```

with:

```text
report_rank
shared_rank
target_id
timestamp
selection_reason=PHASE51_SHARED_TOP_RANK
status.
```

Frozen before render.

---

# 94. No “pretty heatmap” selection

Forbidden:

```text
choose case with sharp diagonal
choose head with clean pattern
choose seed with attractive map.
```

Report cases are chosen entirely by Phase51 ranking.

---

# 95. Figure captions

Every report figure caption should include:

```text
target timestamp
seed/layer/head or grid layout
axis meaning
scale mode
attention-is-temporal caveat
case selection provenance.
```

---

# 96. Mandatory caveat for cross-seed head grids

Caption:

```text
Head indices are shown at matching architectural positions across seeds,
but same-index heads are not assumed to have identical learned semantic roles.
```

---

# 97. Mandatory caveat for attention interpretation

At least once in each report section:

```text
Attention weights describe temporal token-to-token allocation within the Encoder
and are not direct raw-feature importance or causal attribution.
```

---

# 98. Qualitative visual observations allowed

Phase53 may record restrained observations such as:

```text
attention appears broadly diffuse
attention shows a diagonal-like band
attention contains localized source columns
attention contains repeated temporal bands
different heads show visibly distinct patterns.
```

Only if directly visible.

---

# 99. Qualitative observations forbidden

Do not say:

```text
Head 3 is best
Head 2 is useless
this head causes the prediction
the model uses temperature here
this pattern proves daily seasonality
this attention explains the error.
```

Those claims exceed Phase53 evidence.

---

# 100. No subjective morphology label as a hard scientific output

If qualitative notes are recorded, keep them in:

```text
attention_heatmap_visual_notes.csv
```

as optional descriptive notes.

Do not use them for automated ranking or downstream statistical groups.

---

# 101. Quantitative summaries already belong to Phase52/55

If a visual observation needs validation, use:

```text
Phase52 entropy/recent-mass/top-source summaries
```

later in Phase55.

Do not derive ad hoc metrics just to support one heatmap.

---

# 102. Heatmap generation coverage

Required complete coverage:

```text
all dense cases
×
all 3 seeds
×
all layers
×
all heads
```

inside V1 grids.

This guarantees no visually uninteresting head/case is omitted.

---

# 103. Individual map coverage

Standalone per-head images do not need to be generated for every case if V1 grids preserve every panel.

At minimum generate individual maps for:

```text
report-selected shared top5
```

if required for document layout.

---

# 104. Cross-seed grid coverage

Required only for:

```text
report-selected shared top5
```

unless storage/time makes full coverage easy.

Do not redefine case set after rendering.

---

# 105. Fixed-probability coverage

Required for:

```text
report-selected shared top5.
```

Optional for full dense-case catalog.

---

# 106. No baseline heatmaps

Persistence and LSTM do not have this Transformer self-attention artifact.

No attempt to create analogous fake attention heatmaps.

---

# 107. No feature heatmap fabricated from input-feature values

Phase53 is attention-only.

Feature/input plots remain in Phase51 context.

---

# 108. Heatmap visual QA

For every generated image, verify:

```text
file exists
non-zero size
expected width/height
expected panel count
expected title metadata
axis labels present via render metadata
colorbar configured
no exception/warning indicating clipped array.
```

Programmatic image-content OCR is not required.

---

# 109. Panel count audit

V1 per-case/per-seed grid:

\[
Panels=N_{layers}\times N_{heads}.
\]

V2 per-case/per-layer cross-seed grid:

\[
Panels=3\times N_{heads}.
\]

Hard metadata check.

---

# 110. Missing panel policy

A missing head/layer panel:

```text
FAIL.
```

Do not render blank placeholder and continue.

---

# 111. NaN panel policy

Phase52 should have prevented this.

If raw map contains NaN/Inf:

```text
FAIL source integrity
```

not:

```text
render missing color.
```

---

# 112. Scale saturation audit

Canonical Mode A:

```text
no attention value >1
```

except numerical tolerance already handled.

Mode B:

```text
vmax equals actual case max
```

so no clipping by definition.

---

# 113. Zero-max edge case

Impossible for valid row-normalized attention.

If:

```text
case_attention_max <= 0
```

then:

```text
FAIL.
```

Do not create arbitrary vmax.

---

# 114. Empty-case-set edge case

If Phase52 dense case count is zero:

```text
Phase53 FAIL / BLOCKED
```

because heatmap analysis has no scientific case set.

Do not choose new cases.

---

# 115. Layer/head dimension edge case

If final model has:

```text
N_layers=1
or
N_heads=1
```

grid renderer must still work.

No assumption of B0:

```text
2 layers / 4 heads.
```

Runtime final config is authoritative.

---

# 116. Lookback edge case

If final `L` is:

```text
36
72
144
```

tick policy adapts automatically.

No hard-coded 144×144 rendering assumption.

---

# 117. Pooling caveat in titles/report

If final pooling:

```text
LAST_STEP
```

note:

```text
the final regression head consumes the newest encoded token,
so the bottom query row is particularly relevant for Phase54.
```

If:

```text
MEAN
```

note:

```text
prediction pools all encoded positions;
a full-matrix view is therefore important and last-query alone is incomplete.
```

Do not alter heatmap rendering itself.

---

# 118. RevIN caveat

If RN1 active:

```text
attention was produced by the exact RevIN-enabled forward path.
```

No special heatmap transformation required.

Do not claim RevIN makes attention more interpretable.

---

# 119. Error-case context in heatmap catalog

Catalog may include:

```text
shared hardness rank
ALL_UNDER/ALL_OVER/MIXED
EXTREME_HIGH
CHANGE_RAPID
```

for navigation.

These labels are not used to alter scale or panel selection.

---

# 120. No heatmap sorting by regime within canonical case index

Preserve Phase52/51 case order.

Separate filtered views/indexes may be created later without changing raw case ordering.

---

# 121. Heatmap rendering reproducibility

Given:

```text
same raw matrix
same render config
same plotting environment
```

panel mapping should be deterministic.

Byte-identical PNG is not a scientific requirement because image metadata/render backend can vary.

Scientific reproducibility is guaranteed by:

```text
raw SHA
render-config SHA
case-order SHA
position-map SHA
render manifest.
```

---

# 122. Image checksum

Optional but recommended:

```text
SHA256 per generated PNG.
```

Useful for artifact integrity.

Do not use image checksum as substitute for raw source checksum.

---

# 123. Derived image immutability

After official Phase53 rendering:

```text
images = frozen derived artifacts.
```

If visual styling changes later:

```text
create render version v2
```

rather than overwriting without record.

---

# 124. Output directory

```text
artifacts/
└── attention_heatmaps/
    ├── attention_heatmaps_manifest.json
    ├── attention_heatmaps_contract.json
    ├── phase53_preflight_audit.csv
    ├── attention_heatmap_source_verification.csv
    ├── attention_heatmap_render_config.json
    ├── attention_heatmap_render_config_fingerprint.json
    ├── attention_heatmap_orientation_tests.csv
    ├── attention_heatmap_axis_tick_audit.csv
    ├── attention_heatmap_case_scale_manifest.csv
    ├── attention_heatmap_report_cases.csv
    ├── attention_heatmap_catalog.csv
    ├── attention_heatmap_case_index.csv
    ├── attention_heatmap_render_audit.csv
    ├── attention_heatmap_visual_notes.csv
    ├── images/
    │   ├── case_grids/
    │   │   ├── CASE_000_SEED_42_CASE_SHARED_SCALE.png
    │   │   ├── CASE_000_SEED_123_CASE_SHARED_SCALE.png
    │   │   ├── CASE_000_SEED_2026_CASE_SHARED_SCALE.png
    │   │   └── ...
    │   ├── cross_seed_report/
    │   │   ├── CASE_000_LAYER_01_CROSS_SEED_CASE_SHARED_SCALE.png
    │   │   ├── CASE_000_LAYER_02_CROSS_SEED_CASE_SHARED_SCALE.png
    │   │   └── ...
    │   ├── fixed_probability_report/
    │   │   ├── CASE_000_FIXED_PROBABILITY.png
    │   │   └── ...
    │   └── individual_maps/
    │       ├── CASE_000_SEED_42_LAYER_01_HEAD_01_CASE_SHARED_SCALE.png
    │       └── ...
    ├── attention_heatmap_image_checksums.json
    ├── attention_heatmap_findings.csv
    ├── attention_heatmap_tests.csv
    ├── attention_heatmap_discrepancies.json
    ├── phase54_last_query_attention_context_handoff.json
    ├── attention_heatmap_summary.json
    ├── attention_heatmap_catalog.md
    ├── attention_heatmap_report.md
    ├── README_ATTENTION_HEATMAPS.md
    └── phase_53_signoff.json
```

Runtime number of cases/layers/heads must be discovered, not fabricated.

---

# 125. Required outputs

```text
O53.1  Heatmap manifest
O53.2  Heatmap contract
O53.3  Preflight audit
O53.4  Raw-source verification
O53.5  Frozen render config
O53.6  Render-config fingerprint
O53.7  Orientation tests
O53.8  Axis/lag tick audit
O53.9  Case-shared scale manifest
O53.10 Report-selected case manifest
O53.11 Complete V1 case/seed grids
O53.12 V2 cross-seed report grids
O53.13 Mode A fixed-probability report views
O53.14 Optional individual per-head maps
O53.15 Heatmap catalog
O53.16 Case visualization index
O53.17 Render QA audit
O53.18 Optional visual notes
O53.19 Image checksums
O53.20 Findings
O53.21 Phase54 context handoff
O53.22 Tests
O53.23 Discrepancies
O53.24 Summary JSON
O53.25 Catalog Markdown
O53.26 Human-readable report
O53.27 README
O53.28 Sign-off
```

---

# 126. Heatmap manifest

`attention_heatmaps_manifest.json`:

```text
phase=53
version=ATTENTION_HEATMAPS-v1
source_phase52_version
final_lock_sha256
case_selection_contract_sha256
dense_case_order_sha256
relative_position_map_sha256
seed_list=[42,123,2026]
lookback_steps
num_layers
num_heads
case_count
render_modes=[
  FIXED_PROBABILITY,
  CASE_SHARED_SCALE
]
V1_complete_case_grids=true
V2_report_shared_top5=true
new_attention_extraction=false
new_test_inference=false
head_selection=false
seed_selection=false
case_selection_changed=false
attention_feature_importance_claim=false
attention_causal_claim=false
status
created_at
```

---

# 127. Heatmap contract

`attention_heatmaps_contract.json` must freeze:

```text
Raw source:
Phase52 float32 dense attention only.

Per-map axes:
y=query
x=source.

Position order:
0 oldest
L-1 newest.

Display:
oldest→newest left-to-right
oldest→newest top-to-bottom.

Matrix transform:
NONE.

Masks:
no triangle hiding.

Interpolation:
none/nearest.

Color:
sequential attention-weight scale.

Mode A:
0..1.

Mode B:
0..case-wide max across all seeds/layers/heads.

V1:
all dense cases × all seeds
layer-by-head grids.

V2:
Phase51 shared ranks 1–5 only
cross-seed grids by layer.

No:
per-panel auto-scale
percentile clipping
log transform
head averaging as primary
seed averaging
head ranking
case selection changes
causal claims.
```

---

# 128. Preflight audit

`phase53_preflight_audit.csv`:

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
phase53_ready=true
3 dense raw files available
3 raw checksums match
same dense-case order
same case count
same layers
same heads
same lookback
float32
relative-position map verified
case metadata available
Phase51 shared top5 ranks available
render config frozen before official rendering
output directory clean/versioned.
```

---

# 129. Raw-source verification schema

`attention_heatmap_source_verification.csv`:

```text
seed
raw_file
expected_sha256
observed_sha256
shape
dtype
case_count
layer_count
head_count
lookback
finite
row_sum_recheck
case_order_match
status
```

---

# 130. Render config schema

`attention_heatmap_render_config.json`:

```text
version
matrix_orientation=QUERY_ROWS_SOURCE_COLUMNS
transpose=false
origin=ROW0_TOP
x_order=OLDEST_TO_NEWEST
y_order=OLDEST_TO_NEWEST_TOP_TO_BOTTOM
x_label
y_label
lag_tick_steps_requested=[1,6,36,72,144]
include_oldest_tick=true
interpolation=NONE_OR_NEAREST
aspect=SQUARE_PER_MATRIX
colorbar_label=Attention weight
colormap_policy=SEQUENTIAL_PERCEPTUALLY_UNIFORM
mode_A={
  name: FIXED_PROBABILITY,
  vmin: 0,
  vmax: 1
}
mode_B={
  name: CASE_SHARED_SCALE,
  vmin: 0,
  vmax_formula: MAX_OVER_ALL_SEEDS_LAYERS_HEADS_FOR_CASE
}
V1_layout={
  rows: layers,
  columns: heads
}
V2_layout={
  rows: seeds,
  columns: heads
}
report_case_rule=PHASE51_SHARED_RANK_1_TO_5
resolution_policy
figure_size_policy
annotation_policy
created_before_render=true
```

---

# 131. Orientation-test schema

`attention_heatmap_orientation_tests.csv`:

```text
test_case
matrix_shape
query_idx
source_idx
expected_x
expected_y
transpose_applied
origin_policy
diagonal_direction
last_query_display_edge
pass
status
```

Required synthetic and real-source tests.

---

# 132. Axis tick audit schema

`attention_heatmap_axis_tick_audit.csv`:

```text
lookback
requested_lag_steps
included
position_idx0
lag_minutes
x_tick_label
y_tick_label
mapping_source
status
```

---

# 133. Case scale manifest schema

`attention_heatmap_case_scale_manifest.csv`:

```text
case_row_idx0
target_id
target_timestamp
case_raw_min
case_raw_max
seed42_max
seed123_max
seed2026_max
mode_B_vmin
mode_B_vmax
all_panels_share_scale
status
```

---

# 134. Report case manifest schema

`attention_heatmap_report_cases.csv`:

```text
report_order
case_row_idx0
target_id
target_timestamp
shared_worst_rank
selection_rule
regime_labels
status
```

Only Phase51 shared ranks 1–5.

---

# 135. Heatmap catalog schema

`attention_heatmap_catalog.csv`:

```text
image_id
path
image_type
case_row_idx0
target_id
target_timestamp
seed
layer_idx0_if_specific
layer_display_if_specific
head_idx0_if_specific
head_display_if_specific
render_mode
vmin
vmax
grid_rows
grid_columns
panel_count
raw_source_file
raw_source_sha256
case_order_sha256
position_map_sha256
render_config_sha256
report_selected
image_sha256_if_available
status
```

---

# 136. Case index schema

`attention_heatmap_case_index.csv`:

```text
case_row_idx0
target_id
target_timestamp
selection_roles
shared_rank
target_level_regime
extreme_high_regime
change_magnitude_regime
change_direction_regime
time_of_day_regime
day_type_regime
seed42_grid
seed123_grid
seed2026_grid
cross_seed_layer_grids
fixed_probability_view
status
```

---

# 137. Render QA schema

`attention_heatmap_render_audit.csv`:

```text
image_id
file_exists
file_size_bytes
expected_panel_count
observed_panel_count
orientation_verified
axis_semantics_verified
scale_mode
vmin
vmax
raw_max_within_scale
colorbar_configured
interpolation_policy_match
render_complete
status
```

---

# 138. Visual notes schema

Optional:

`attention_heatmap_visual_notes.csv`:

```text
case_row_idx0
target_id
seed
layer
head
note
note_scope=QUALITATIVE_DESCRIPTIVE_ONLY
used_for_selection=false
status
```

No numeric rank.

---

# 139. Image checksum artifact

`attention_heatmap_image_checksums.json`:

```text
render_config_sha256
images={
  image_id: sha256,
  ...
}
status
```

Optional if image count is very large, but recommended.

---

# 140. Heatmap catalog Markdown

`attention_heatmap_catalog.md` should organize images by:

```text
Case
  Seed42
  Seed123
  Seed2026
  Cross-seed report views if applicable
```

Not by:

```text
“best-looking head”.
```

---

# 141. Main report heatmap selection

Main coursework/report should prioritize:

```text
shared worst ranks 1–5
```

because those cases were selected before attention inspection and are cross-seed hard cases.

This gives defensible attention examples.

---

# 142. Report figure density

Do not insert dozens of tiny panels into the main report.

Suggested report strategy:

```text
1–2 representative cross-seed grids
from deterministic shared ranks
+
appendix/catalog for complete heatmap set.
```

If space is limited, use:

```text
shared rank1 first
then rank2...
```

not visual cherry-picking.

---

# 143. “Representative” wording caution

If only shared rank1 appears in the body, call it:

```text
highest-ranked shared worst-error case
```

not:

```text
representative case
```

unless representation criteria are separately defined.

---

# 144. No publication figure manually altered

Do not:

```text
crop out unfavorable heads
adjust contrast head-by-head
erase empty-looking regions
manually annotate attention peaks not in source.
```

Any annotation must be generated from deterministic metadata.

---

# 145. Qualitative interpretation protocol

Allowed workflow:

```text
render first
freeze image catalog
then write qualitative notes.
```

Do not:

```text
write desired story
then alter rendering to support it.
```

---

# 146. Visual observation hierarchy

Phase53 may describe:

```text
1. Global map structure
2. Layer-to-layer visible differences
3. Head-to-head visible diversity
4. Case-specific temporal patterns
```

but without ranking or causal claims.

---

# 147. Global map structure wording

Safe:

> This head shows a visually concentrated band around a subset of source positions.

Safe:

> The attention distribution appears diffuse across many historical source positions.

Safe:

> A diagonal-like structure is visible across several query positions.

Unsafe:

> This is the most useful head.

---

# 148. Layer wording

Safe:

> The later encoder layer exhibits a visibly different allocation pattern from the earlier layer for this case.

Unsafe:

> Layer 2 understands long-term dependencies better.

Quantitative comparison belongs Phase55.

---

# 149. Cross-seed wording

Safe:

> The same-index head maps differ visually across seeds.

Must add caveat:

```text
same-index heads are not guaranteed to share the same semantic role.
```

---

# 150. Case-error wording

Safe:

> This heatmap corresponds to the highest-ranked shared worst-error case defined in Phase51.

Unsafe:

> This attention pattern caused the worst error.

---

# 151. Feature wording

Forbidden:

```text
the model attends to T1
the model ignores humidity
```

because attention positions correspond to timestamp tokens after feature projection.

---

# 152. Regime wording

Safe:

> This case was classified as EXTREME_HIGH and CHANGE_RAPID using Train-defined Phase50 thresholds.

Unsafe:

> The extreme-high regime caused this attention structure.

---

# 153. Heatmap not a metric

Do not create:

```text
attention heatmap score
```

in Phase53.

Quantitative attention metrics were extracted in Phase52 and will be compared in Phase55/56.

---

# 154. No model-quality judgment from “sharp” vs “diffuse”

Sharp attention is not inherently good.

Diffuse attention is not inherently bad.

No Phase53 PASS/FAIL based on visual morphology.

---

# 155. Phase54 context handoff

Phase54 remains anchored to:

```text
Phase52 last_query_attention_seed*.npz
```

not images.

Phase53 provides only visual context/index.

---

# 156. Phase54 context handoff schema

`phase54_last_query_attention_context_handoff.json`:

```text
source_phase53_version
source_phase52_version
final_lock_sha256
heatmap_render_config_sha256
dense_case_order_sha256
report_case_manifest
heatmap_catalog
case_index
canonical_orientation={
  y: QUERY,
  x: SOURCE,
  oldest_position: 0,
  newest_position: L-1,
  last_query_display_edge: BOTTOM
}
last_query_source=PHASE52_RAW_LAST_QUERY
do_not_extract_last_query_from_png=true
ready_for_phase54_context=true
```

---

# 157. Phase54 must not digitize bottom heatmap row

Although bottom row visually corresponds to last query, numerical Phase54 work must read:

```text
Phase52 last-query NPZ
```

not image pixels.

---

# 158. Discrepancy taxonomy

`attention_heatmap_discrepancies.json`:

```text
PHASE52_NOT_APPROVED
PHASE53_HANDOFF_NOT_READY
RAW_DENSE_FILE_MISSING
RAW_SHA_MISMATCH
RAW_DTYPE_MISMATCH
RAW_SHAPE_MISMATCH
CASE_ORDER_MISMATCH
CASE_COUNT_MISMATCH
HEAD_COUNT_MISMATCH
LAYER_COUNT_MISMATCH
LOOKBACK_MISMATCH
POSITION_MAP_MISMATCH
RENDER_CONFIG_NOT_FROZEN
MATRIX_TRANSPOSED
QUERY_SOURCE_AXES_SWAPPED
ROW_ORIGIN_REVERSED
TEMPORAL_ORDER_REVERSED
LAST_QUERY_NOT_BOTTOM
LAG_TICK_MAPPING_ERROR
TARGET_TOKEN_ADDED_TO_AXIS
CAUSAL_TRIANGLE_HIDDEN
ATTENTION_VALUES_RENORMALIZED
ATTENTION_VALUES_SMOOTHED
ATTENTION_VALUES_THRESHOLDed
ATTENTION_VALUES_LOG_TRANSFORMED
PERCENTILE_CLIPPING_USED_AS_CANONICAL
PER_PANEL_AUTOSCALE_USED_IN_COMPARISON_GRID
CASE_SHARED_SCALE_INCONSISTENT
FIXED_PROBABILITY_SCALE_NOT_0_1
DIVERGING_COLORMAP_USED
COLORBAR_MISLABELED_AS_IMPORTANCE
INTERPOLATION_USED
MATRIX_RESAMPLED
MISSING_LAYER_PANEL
MISSING_HEAD_PANEL
CASE_SKIPPED_AFTER_VISUAL_INSPECTION
REPORT_CASE_REPLACED_AFTER_VISUAL_INSPECTION
BEST_HEAD_SELECTED
BEST_SEED_SELECTED
SEED_AVERAGED_HEATMAP_CREATED_AS_PRIMARY
HEAD_AVERAGED_HEATMAP_REPLACED_RAW_HEADS
SAME_HEAD_INDEX_ASSUMED_SEMANTICALLY_ALIGNED
NEW_ATTENTION_EXTRACTION_ATTEMPT
NEW_TEST_INFERENCE_ATTEMPT
ATTENTION_MISLABELED_AS_FEATURE_IMPORTANCE
ATTENTION_MISLABELED_AS_CAUSAL_ATTRIBUTION
IMAGE_USED_AS_NUMERIC_SOURCE
OTHER
```

---

# 159. Status model

## PASS

```text
all raw sources/checksums verified
orientation contract verified
render config frozen
all dense cases rendered
all seeds/layers/heads represented
case-shared scales consistent
shared top5 report views generated
absolute probability references generated
catalog/provenance complete
no re-extraction
no case/head/seed selection
Phase54 context ready.
```

## PASS_WITH_WARNING

Possible:

```text
very dense maps difficult to visually inspect
fixed 0..1 view visually compressed
optional individual maps omitted for non-report cases
image file volume large
minor rendering-environment drift.
```

These are visualization limitations, not scientific failure.

## FAIL

Examples:

```text
transpose/orientation error
raw checksum mismatch
per-panel autoscaling in comparison grids
missing heads
case cherry-pick
new extraction
attention labeled feature importance.
```

---

# 160. Execution sequence

```text
1. Verify Phase52 signoff/handoff.
2. Verify 3 dense raw attention files/checksums.
3. Verify same case/layer/head/lookback structure.
4. Freeze render config and render-config SHA.
5. Run synthetic orientation unit test.
6. Run real-source orientation/lag audit.
7. Build report-selected shared top5 manifest from Phase51 ranks.
8. Compute CASE_SHARED_SCALE vmax for every dense case.
9. Freeze case-scale manifest.
10. Render V1 case×seed layer/head grids for all dense cases.
11. Render V2 cross-seed layer grids for shared top5.
12. Render FIXED_PROBABILITY views for shared top5.
13. Render optional individual per-head report files.
14. Build heatmap catalog.
15. Build case visualization index.
16. Run panel/scale/orientation render QA.
17. Generate image checksums if enabled.
18. Write restrained qualitative notes only after catalog freeze.
19. Write findings/report/catalog Markdown.
20. Write Phase54 context handoff.
21. Run tests/discrepancy audit.
22. Sign off.
```

---

# 161. Recommended rendering pseudocode

```text
p52 = load_phase52_signoff()
assert p52.overall_status in {"PASS","PASS_WITH_WARNING"}

handoff = load_phase53_handoff()
assert handoff.ready_for_phase53

raw = {
    42:   load_dense_attention_seed42(),
    123:  load_dense_attention_seed123(),
    2026: load_dense_attention_seed2026()
}

verify_raw_checksums(raw)
verify_same_dense_case_order(raw)
verify_same_shapes(raw)

render_cfg = freeze_render_config(
    transpose=False,
    y_axis="QUERY",
    x_axis="SOURCE",
    row0_at_top=True,
    x_order="OLDEST_TO_NEWEST",
    y_order="OLDEST_TO_NEWEST_TOP_TO_BOTTOM",
    interpolation="NONE",
    mode_A={"vmin":0,"vmax":1},
    mode_B="CASE_SHARED_MAX",
    report_cases="PHASE51_SHARED_RANK_1_TO_5"
)

run_synthetic_orientation_test(render_cfg)
run_real_source_orientation_test(raw, render_cfg)

report_cases = resolve_phase51_shared_ranks(1,5)
freeze_report_case_manifest(report_cases)

case_scales = {}

for case_idx in all_dense_case_indices:

    case_values = concatenate(
        raw[42][case_idx],
        raw[123][case_idx],
        raw[2026][case_idx]
    )

    vmax_case = max(case_values)
    assert vmax_case > 0

    case_scales[case_idx] = {
        "vmin":0,
        "vmax":vmax_case
    }

freeze_case_scale_manifest(case_scales)

for case_idx in all_dense_case_indices:

    for seed in [42,123,2026]:

        A_case = raw[seed][case_idx]
        # shape [Layers,Heads,L,L]

        render_layer_head_grid(
            A_case,
            rows="LAYERS",
            columns="HEADS",
            matrix_transform="NONE",
            scale=case_scales[case_idx],
            axis_map=relative_position_map,
            colorbar_label="Attention weight",
            mode="CASE_SHARED_SCALE"
        )

for case_idx in report_cases:

    # cross-seed grid for each layer
    for layer_idx in range(N_layers):

        panels = [
            raw[seed][case_idx,layer_idx,:,:,:]
            for seed in [42,123,2026]
        ]

        render_cross_seed_grid(
            panels,
            rows=[42,123,2026],
            columns="HEADS",
            shared_scale=case_scales[case_idx]
        )

    render_fixed_probability_reference(
        case_idx,
        raw,
        vmin=0,
        vmax=1
    )

    render_optional_individual_head_maps(...)

build_heatmap_catalog()
build_case_index()

run_render_QA(
    expected_V1_panels=N_layers*N_heads,
    expected_V2_panels=3*N_heads,
    orientation="QUERY_ROWS_SOURCE_COLUMNS",
    no_interpolation=True
)

freeze_derived_images()

write_phase54_context_handoff(
    numeric_last_query_source="PHASE52_RAW_NPZ",
    do_not_digitize_images=True
)

assert no_new_attention_extraction
assert no_case_selection_change
assert no_head_selection
assert no_seed_selection
assert no_feature_importance_claim

signoff_phase53()
```

---

# 162. Preflight acceptance checklist

```text
[ ] Phase52 PASS/PASS_WITH_WARNING.
[ ] phase53_ready=true.
[ ] Seed42 dense NPZ exists.
[ ] Seed123 dense NPZ exists.
[ ] Seed2026 dense NPZ exists.
[ ] Raw SHA256s match.
[ ] dtype float32.
[ ] Same dense case count.
[ ] Same dense case order.
[ ] Same layer count.
[ ] Same head count.
[ ] Same lookback.
[ ] Relative position map verified.
[ ] Case metadata available.
[ ] Phase51 shared-rank data available.
[ ] No new inference/extraction required.
```

---

# 163. Render-contract acceptance checklist

```text
[ ] Render config written before official rendering.
[ ] Render config SHA generated.
[ ] Matrix transpose=false.
[ ] x=source.
[ ] y=query.
[ ] Row0 top.
[ ] Position0 oldest.
[ ] Position L-1 newest.
[ ] Oldest→newest left-to-right.
[ ] Oldest→newest top-to-bottom.
[ ] Last-query row visually bottom.
[ ] No target token added.
[ ] No causal triangle hidden.
[ ] Interpolation disabled.
[ ] Sequential color semantics.
[ ] Colorbar label Attention weight.
```

---

# 164. Scale acceptance checklist

```text
[ ] Mode A vmin=0.
[ ] Mode A vmax=1.
[ ] Mode B computed once per case.
[ ] Mode B max uses all 3 seeds/layers/heads for that case.
[ ] Same Mode B scale applied to all panels of case.
[ ] No per-panel auto-scale in comparison grid.
[ ] No percentile clipping.
[ ] No log scale.
[ ] No normalization.
[ ] Raw max never clipped in canonical view.
[ ] Scale mode recorded in every image manifest row.
```

---

# 165. Orientation acceptance checklist

```text
[ ] Synthetic asymmetric-matrix test passes.
[ ] Real-source coordinate mapping test passes.
[ ] Diagonal is top-left→bottom-right.
[ ] Last-query row is bottom.
[ ] Source newest position is right edge.
[ ] Query newest position is bottom edge.
[ ] Lag ticks match Phase52 map.
[ ] No axis reversal.
```

---

# 166. V1 coverage acceptance checklist

```text
[ ] Every dense case rendered.
[ ] Seed42 grid for every case.
[ ] Seed123 grid for every case.
[ ] Seed2026 grid for every case.
[ ] Rows=layers.
[ ] Columns=heads.
[ ] Panel count=N_layers*N_heads.
[ ] No missing head.
[ ] No missing layer.
[ ] One shared colorbar.
[ ] Mode=CASE_SHARED_SCALE.
```

---

# 167. V2/report acceptance checklist

```text
[ ] Report cases exactly Phase51 shared ranks1–5.
[ ] No visual case substitution.
[ ] Cross-seed grid generated per report case/layer.
[ ] Rows=42,123,2026.
[ ] Columns=heads.
[ ] Panel count=3*N_heads.
[ ] Same case-shared scale.
[ ] Same-index head semantic caveat included.
[ ] FIXED_PROBABILITY reference generated.
```

---

# 168. Catalog/provenance acceptance checklist

```text
[ ] Every image has catalog row.
[ ] Every image maps to raw file SHA.
[ ] Case row index stored.
[ ] Target ID stored.
[ ] Seed stored.
[ ] Layer/head scope stored.
[ ] Render mode stored.
[ ] vmin/vmax stored.
[ ] Position-map SHA stored.
[ ] Render-config SHA stored.
[ ] Report-selected flag stored.
[ ] Optional image SHA stored.
```

---

# 169. Interpretation-scope acceptance checklist

```text
[ ] No head ranked.
[ ] No seed ranked.
[ ] No case changed after viewing maps.
[ ] No attention-based case addition.
[ ] No error-conditioned statistics.
[ ] No last-query quantitative analysis.
[ ] No seed-averaged heatmap.
[ ] Attention labeled temporal.
[ ] No raw-feature importance claim.
[ ] No causal claim.
[ ] Same-index heads across seeds not assumed semantically identical.
```

---

# 170. Acceptance criteria

Phase53 PASS only when:

```text
All Phase52 dense raw attention files are verified by checksum and retain the exact same frozen case ordering across the three final seeds.

The rendering contract is frozen before official heatmap generation.

Each raw [L,L] matrix is rendered without transposition, with query on the y-axis and source/key on the x-axis.

Position 0 is displayed as the oldest historical input and position L-1 as the newest historical input.

The query axis progresses from oldest at the top to newest at the bottom, making the last-query row the bottom row.

The source axis progresses from oldest on the left to newest on the right.

Lag ticks are derived from the frozen Phase52 relative-position map and are not manually approximated.

No forecast-target row/column or causal-mask triangle is introduced.

Raw attention values are rendered without renormalization, smoothing, thresholding, log transformation or numerical resampling.

The FIXED_PROBABILITY view uses the exact [0,1] probability scale.

The CASE_SHARED_SCALE view uses one frozen maximum per target case computed across all three seeds, all layers and all heads, preventing per-panel auto-scale distortion.

Every frozen dense case is rendered for all three seeds with all layers and all heads represented.

The deterministic Phase51 shared worst-error ranks 1–5 are used for report-oriented cross-seed grids; no case is selected because its attention pattern is visually appealing.

Cross-seed grids explicitly warn that same-index heads are not guaranteed to have the same learned semantic role.

All images are traceable to raw attention checksums, case ordering, position mappings and render-config fingerprint.

Heatmap images are treated only as derived visualizations; downstream numerical analysis continues to use Phase52 raw arrays.

No new Test inference, attention extraction, best-head selection, best-seed selection, feature-importance claim or causal explanation occurs.

Phase54 receives visualization context while its numerical last-query source remains the frozen Phase52 NPZ artifacts.
```

---

# 171. Failure conditions

Phase53 FAIL if:

```text
raw source checksum mismatches

heatmap matrix is transposed

query/source axes are reversed

oldest/newest order is reversed

last-query row is not at the documented edge

lag labels are wrong

target token is added

upper triangle is hidden as if causal mask existed

attention is smoothed/interpolated

raw values are renormalized

per-panel autoscaling is used in official comparison grids

percentile clipping hides values in canonical views

Mode A is not exactly 0..1

case-shared scale differs by seed/head for one case

a layer/head panel is omitted

report case is swapped after visual inspection

only attractive heads are shown

a seed is omitted

seed-averaged attention is presented as primary

new attention is extracted

images are used as numerical source

attention is described as raw-feature importance

attention is claimed to causally explain prediction/error

same-index heads are assumed semantically equivalent across seeds.
```

---

# 172. Common mistakes

## 172.1 Vẽ `attention.T`

Sai orientation nếu không đổi labels tương ứng.

Canonical:

```text
rows=query
columns=source.
```

## 172.2 Dùng y-axis source và x-axis query nhưng caption vẫn ghi ngược

Sai semantic labeling.

## 172.3 Để plotting library tự đưa row0 xuống dưới

Sai project orientation nếu metadata không đổi.

Canonical:

```text
oldest query top
newest query bottom.
```

## 172.4 Hide upper triangle vì nghĩ Transformer cần causal mask

Sai. Final Encoder không causal-mask historical inputs.

## 172.5 Auto-scale từng head

Làm intensity giữa heads không còn so sánh trực tiếp được.

## 172.6 p99 clipping để heatmap đẹp hơn nhưng không ghi

Sai scientific visualization.

## 172.7 Average tất cả heads rồi chỉ show một map

Mất head-specific information.

## 172.8 Average cùng head index qua 3 seeds

Chưa hợp lệ vì head semantic permutation có thể xảy ra.

## 172.9 Chọn 3 heatmaps đẹp nhất để đưa report

Sai cherry-pick.

Dùng:

```text
Phase51 shared ranks 1–5.
```

## 172.10 Gọi màu sáng là “feature quan trọng”

Sai. Đây là temporal token attention.

## 172.11 Đọc bottom row từ PNG để tính last-query metrics

Sai. Phase54 dùng raw Phase52 NPZ.

## 172.12 Map diffuse rồi kết luận head vô dụng

Sai. Sharpness không đồng nghĩa quality.

---

# 173. Human-readable report structure

`attention_heatmap_report.md`:

```text
1. Objective
2. Frozen Phase52 attention sources
3. Query/source tensor semantics
4. Temporal position and lag mapping
5. Rendering contract
6. Color-scale methodology
7. FIXED_PROBABILITY vs CASE_SHARED_SCALE
8. Full case/seed heatmap coverage
9. Shared worst-case rank1 heatmaps
10. Shared worst-case ranks2–5 heatmaps
11. Layer/head visual diversity
12. Cross-seed visual comparison caveat
13. Qualitative visual observations
14. Why heatmaps are not feature importance
15. Why heatmaps are not causal explanation
16. Why images are not numerical source
17. Handoff to last-query attention analysis
18. Limitations
19. Definition of Done
```

---

# 174. README requirements

`README_ATTENTION_HEATMAPS.md` explains:

```text
what each heatmap cell means
why y=query and x=source
why row0 is oldest
why last-query is bottom row
how lag ticks map to positions
why upper triangle is valid
what FIXED_PROBABILITY scale means
what CASE_SHARED_SCALE means
why per-panel autoscaling is avoided
why no smoothing/interpolation is used
why report cases are Phase51-defined
why same-index heads across seeds may not share function
why attention is temporal rather than raw-feature importance
why numerical downstream work must use Phase52 raw arrays.
```

---

# 175. Summary artifact

`attention_heatmap_summary.json`:

```text
version
source_phase52_version
final_lock_sha256
case_selection_contract_sha256
render_config_sha256
seed_list
dense_case_count
lookback
layers
heads
raw_attention_sha256s
orientation={
  y: QUERY,
  x: SOURCE,
  row0: OLDEST,
  row_last: NEWEST,
  last_query_edge: BOTTOM,
  source_left: OLDEST,
  source_right: NEWEST
}
render_modes
case_shared_scale_count
report_case_count
V1_grid_count
V2_grid_count
fixed_probability_view_count
catalog_row_count
orientation_tests_passed
scale_audits_passed
all_cases_covered
all_heads_covered
all_layers_covered
new_attention_extraction=false
best_head_selected=false
best_seed_selected=false
case_selection_changed=false
attention_feature_importance_claim=false
attention_causal_claim=false
phase54_context_ready
overall_status
```

Do not fabricate runtime counts before rendering.

---

# 176. Phase53 sign-off

`phase_53_signoff.json` minimum:

```text
phase=53
phase_name=Attention heatmaps
version=ATTENTION_HEATMAPS-v1
source_phase52_version
final_lock_sha256
case_selection_contract_sha256
render_config_sha256
seed_list=[42,123,2026]
dense_case_count
lookback_steps
num_layers
num_heads
orientation=QUERY_ROWS_SOURCE_COLUMNS
row0=OLDEST
last_query_edge=BOTTOM
fixed_probability_scale_verified
case_shared_scale_verified
orientation_tests_verified
lag_tick_mapping_verified
all_dense_cases_rendered
all_seeds_rendered
all_layers_rendered
all_heads_rendered
report_shared_top5_rendered
catalog_complete
new_attention_extraction=false
new_test_inference=false
head_selection=false
seed_selection=false
case_selection_changed=false
attention_feature_importance_claim=false
attention_causal_claim=false
phase54_context_ready
warnings
overall_status
created_at
```

---

# 177. Definition of Done

\[
\boxed{
Verified\ Raw\ Dense\ Attention
+
Correct\ Query/Source\ Orientation
+
Correct\ Temporal\ Order
+
Transparent\ Lag\ Mapping
+
Fixed\ Probability\ View
+
Case\text{-}Shared\ Comparison\ View
+
All\ Cases/Seeds/Layers/Heads
+
Deterministic\ Report\ Cases
+
Complete\ Catalog
+
No\ Re\text{-}Extraction
+
No\ Head\ Selection
+
No\ Causal\ Claim
}
\]

---

# 178. Final status contract

```text
PHASE 53 visualizes frozen dense attention.

Source:
Phase52 raw float32 dense NPZ only.

Matrix:
[row,column]
=
[query,source].

Display:
x = source
y = query.

Temporal order:
position0 oldest
position L-1 newest.

Screen layout:
source oldest→newest left→right
query oldest→newest top→bottom.

Last-query:
bottom row.
Do not quantify from image.

Modes:

A. FIXED_PROBABILITY
vmin=0
vmax=1
global absolute reference.

B. CASE_SHARED_SCALE
vmin=0
vmax=max attention across
all seeds/layers/heads for one target case.
Comparable within case.

Forbidden:
per-panel canonical auto-scale
percentile clipping
log transform
smoothing
interpolation
renormalization
transpose
causal triangle hiding
seed averaging
head selection
case cherry-picking.

V1:
every dense case
× every seed
grid rows=layers, columns=heads.

V2:
Phase51 shared ranks1–5
cross-seed grid by layer
rows=seeds, columns=heads.

Images:
derived only.
Raw NPZ remains numerical source.

Interpretation:
temporal token-to-token allocation
not feature importance
not causal attribution.

After ATTENTION_HEATMAPS-v1 PASS:
proceed to
PHASE 54 — Last-Query Attention.
```

---

# 179. Final check

Correct:

```text
verify Phase52 raw attention
→ freeze orientation/render contract
→ test query/source orientation
→ freeze case-shared scales
→ render every case/seed/layer/head
→ deterministic shared-top5 report grids
→ catalog + provenance
→ Phase54 visual context handoff
```

Incorrect:

```text
load model
→ extract attention again
```

Incorrect:

```text
auto-scale every head independently
→ visually compare brightness
```

Incorrect:

```text
hide upper triangle
because “future attention = leakage”
```

Incorrect:

```text
pick heatmaps that look convincing
for the report
```

Incorrect:

```text
bright cell
→ call raw feature important
```

Chỉ sau khi:

```text
ATTENTION_HEATMAPS-v1 = PASS / PASS_WITH_WARNING
```

và:

```text
phase54_context_ready = true
```

mới chuyển sang **PHASE 54 — Last-Query Attention**.
