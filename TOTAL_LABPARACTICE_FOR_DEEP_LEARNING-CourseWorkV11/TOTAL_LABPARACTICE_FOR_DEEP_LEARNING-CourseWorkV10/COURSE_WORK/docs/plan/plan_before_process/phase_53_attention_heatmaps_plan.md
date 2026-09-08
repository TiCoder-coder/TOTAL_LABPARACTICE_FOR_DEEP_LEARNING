# Phase 53 — Attention Heatmaps — Pre-Process Plan

**Version:** ATTENTION_HEATMAPS-v1
**Phase ID:** `PHASE_53_ATTENTION_HEATMAPS`
**Architecture amendment:** v1.15 (2026-09-04) — see `docs/RULE_BASE/architecture_rule.md` §7.37.14 and §28
**Status:** APPROVED (Human approval embedded in this prompt per the §7.37.14 amendment activation rules; plan faithfully maps canonical Phase 53 detail)

---

## 1. Scope

Phase 53 visualizes the frozen Phase 52 raw dense attention for the canonical Phase51 dense-case set (44 unique targets × 3 seeds × 2 layers × 4 heads × 72 source × 72 query), plus a Mode A FIXED_PROBABILITY absolute-reference view, plus cross-seed grids for the deterministic Phase51 W2 SHARED_WORST ranks 1–5 report set. No new attention is extracted. No model is loaded. No Test inference is run.

This plan follows the canonical Phase 53 detail `docs/plan-doc/plan_detail_for_each_phase/Phase_53_Attention_heatmaps.md` verbatim in scope and outputs. The Human approval embedded in this prompt constitutes the §7.37.14 plan approval, provided the plan matches the canonical detail — which it does.

---

## 2. Frozen inputs (READ-ONLY)

| Artifact | Source | SHA-locked |
|---|---|---|
| `dense_case_attention_seed42.npz` | `artifacts/attention_extraction/raw/` | yes |
| `dense_case_attention_seed123.npz` | `artifacts/attention_extraction/raw/` | yes |
| `dense_case_attention_seed2026.npz` | `artifacts/attention_extraction/raw/` | yes |
| `attention_dense_case_order.csv` | `artifacts/attention_extraction/` | yes |
| `attention_relative_position_map.csv` | `artifacts/attention_extraction/` | yes |
| `attention_case_position_map.csv` | `artifacts/attention_extraction/` | yes |
| `attention_case_metadata.csv` | `artifacts/attention_extraction/` | yes |
| `raw_attention_checksums.json` | `artifacts/attention_extraction/` | yes |
| `phase_52_signoff.json` | `artifacts/attention_extraction/` | yes |
| `phase53_attention_heatmaps_handoff.json` | `artifacts/attention_extraction/` | yes |
| `worst_shared_top20.csv` | `artifacts/worst_error_analysis/` | yes (Phase51 frozen) |
| `casebook_index.csv` | `artifacts/worst_error_analysis/` | yes (Phase51 frozen) |

All Phase 47/48/49/50/51/52 canonical artifacts are immutable.

---

## 3. Phase 53 sub-phase gate plan

Phase 53 is structured as sub-phases A through H. Sub-phase H is **deferred**; phases A–G execute in this run.

| Sub-phase | Scope |
|---|---|
| **A** | Governance / source / architecture audit + pre-process plan |
| **B** | Phase 53 infrastructure package (read-only rendering module) + render-config freeze |
| **C** | Orientation / axis / lag tests + case-shared scale manifest + report case manifest |
| **D** | V1 rendering (132 case/seed grids), V2 rendering (cross-seed grids), Mode A rendering, V3 rendering |
| **E** | Catalog + case index + render QA + image checksums + (optional) visual notes |
| **F** | Phase 54 context handoff + tests + discrepancies + summary JSON + catalog Markdown |
| **G** | Final attention_heatmap_report.md + README_ATTENTION_HEATMAPS.md + Phase 53 signoff + processing log |
| **H** | **DEFERRED** — notebook presentation-only dashboard (Human approval required) |

---

## 4. Canonical contract (lock)

```text
Raw source:        Phase52 float32 dense NPZ only (no model checkpoint; no re-extraction)
Per-map axes:      y=query, x=source
Position order:    position 0 = oldest, position L-1 = newest
Display:           source oldest→newest left→right
                   query oldest→newest top→bottom
Matrix transform:  NONE (no transpose)
Masks:             none (no causal triangle hiding)
Interpolation:     none / nearest
Color:             sequential, perceptually uniform
Colorbar label:    "Attention weight"
Mode A scale:      [0, 1] (FIXED_PROBABILITY)
Mode B scale:      [0, case-wide max across 3 seeds × L layers × H heads × q × s]
V1:                all 44 dense cases × all 3 seeds; rows=layers, columns=heads
V2:                shared ranks 1–5; rows=seeds, columns=heads; per layer
V3 (optional):     shared top5 individual per-head files
Forbidden:         per-panel autoscale / percentile clipping / log / seed averaging / head averaging as primary / head ranking / case cherry-pick / causal claim / feature importance claim
```

---

## 5. Render policy

```text
Layout V1          rows=layers, columns=heads     panel_count = N_layers × N_heads
Layout V2          rows=seeds (42/123/2026), columns=heads (1..H)     panel_count = 3 × N_heads
Image format       primary PNG, 300 DPI equivalent
Interpolation      NONE / NEAREST
Colormap           perceptually uniform sequential (viridis-equivalent)
Colorbar label     "Attention weight"
Aspect             SQUARE_PER_MATRIX (each panel L×L)
Figure size        determined by rows × columns (consistent across same grid type)
Origin             row0 = TOP
```

---

## 6. Output inventory (must all PASS)

| ID | Artifact | Path |
|----|----------|------|
| O53.1 | heatmap manifest | `artifacts/attention_heatmaps/attention_heatmaps_manifest.json` |
| O53.2 | heatmap contract | `artifacts/attention_heatmaps/attention_heatmaps_contract.json` |
| O53.3 | preflight audit | `artifacts/attention_heatmaps/phase53_preflight_audit.csv` |
| O53.4 | raw-source verification | `artifacts/attention_heatmaps/attention_heatmap_source_verification.csv` |
| O53.5 | frozen render config | `artifacts/attention_heatmaps/attention_heatmap_render_config.json` |
| O53.6 | render-config fingerprint | `artifacts/attention_heatmaps/attention_heatmap_render_config_fingerprint.json` |
| O53.7 | orientation tests | `artifacts/attention_heatmaps/attention_heatmap_orientation_tests.csv` |
| O53.8 | axis tick audit | `artifacts/attention_heatmaps/attention_heatmap_axis_tick_audit.csv` |
| O53.9 | case-shared scale manifest | `artifacts/attention_heatmaps/attention_heatmap_case_scale_manifest.csv` |
| O53.10 | report-selected case manifest | `artifacts/attention_heatmaps/attention_heatmap_report_cases.csv` |
| O53.11 | complete V1 case×seed grids | `artifacts/attention_heatmaps/images/case_grids/*.png` |
| O53.12 | V2 cross-seed report grids | `artifacts/attention_heatmaps/images/cross_seed_report/*.png` |
| O53.13 | Mode A fixed-probability views | `artifacts/attention_heatmaps/images/fixed_probability_report/*.png` |
| O53.14 | optional V3 individual maps | `artifacts/attention_heatmaps/images/individual_maps/*.png` |
| O53.15 | heatmap catalog | `artifacts/attention_heatmaps/attention_heatmap_catalog.csv` |
| O53.16 | case visualization index | `artifacts/attention_heatmaps/attention_heatmap_case_index.csv` |
| O53.17 | render QA | `artifacts/attention_heatmaps/attention_heatmap_render_audit.csv` |
| O53.18 | optional visual notes | `artifacts/attention_heatmaps/attention_heatmap_visual_notes.csv` |
| O53.19 | image checksums | `artifacts/attention_heatmaps/attention_heatmap_image_checksums.json` |
| O53.20 | findings | `artifacts/attention_heatmaps/attention_heatmap_findings.csv` |
| O53.21 | Phase 54 context handoff | `artifacts/attention_heatmaps/phase54_last_query_attention_context_handoff.json` |
| O53.22 | tests | `tests/unit/test_phase53_*.py` |
| O53.23 | discrepancies | `artifacts/attention_heatmaps/attention_heatmap_discrepancies.json` |
| O53.24 | summary JSON | `artifacts/attention_heatmaps/attention_heatmap_summary.json` |
| O53.25 | catalog Markdown | `artifacts/attention_heatmaps/attention_heatmap_catalog.md` |
| O53.26 | main report (19-section) | `artifacts/attention_heatmaps/attention_heatmap_report.md` |
| O53.27 | README | `artifacts/attention_heatmaps/README_ATTENTION_HEATMAPS.md` |
| O53.28 | Phase 53 signoff | `artifacts/attention_heatmaps/phase_53_signoff.json` |

Required outputs (non-optional): O53.1–13, 15–17, 19–28 (24 of 28).
Optional outputs: O53.14 (V3 individual maps), O53.18 (visual notes).
**Phase 53 must PASS only when all REQUIRED outputs exist and verify.**

---

## 7. Source package plan (read-only rendering)

```text
src/course_work/phase53/
├── __init__.py
├── sources.py           # frozen constant + path declarations, lock resolver
├── raw_verify.py        # raw NPZ SHA + shape/dtype/order/position/lag recheck
├── orientation.py       # synthetic + real-source orientation tests
├── scales.py            # Mode-B case-shared max computation
├── render_config.py     # render-config freeze + fingerprint
├── rendering.py         # pure-matplotlib grids + figure writers (no model code)
├── catalog.py           # catalog + case index writers
├── qa.py                # render QA + image checksums
├── report_cases.py      # Phase51 W2 SHARED_WORST ranks 1–5 resolver
├── handoff_phase54.py   # Phase54 context handoff writer
├── writeup.py           # report, README, catalog Markdown, summary, signoff
└── orchestrator.py      # deterministic end-to-end runner (53-B → 53-G)
```

Every module is **read-only** w.r.t. all canonical artifacts.

---

## 8. Hard gates (PASS conditions)

```text
[ ] Phase52 PASS or PASS_WITH_WARNING
[ ] phase53_ready = true
[ ] all 3 raw dense NPZ SHA256 match
[ ] dtype = float32
[ ] same dense case order across seeds
[ ] same L / L / H across seeds
[ ] relative-position map verified
[ ] Phase51 shared-rank data available
[ ] render config frozen BEFORE any official render
[ ] synthetic + real-source orientation tests PASS
[ ] axis tick audit PASS
[ ] case-shared scales computed before any official render
[ ] all required O53.1–28 outputs present and PASS
[ ] no model checkpoint load
[ ] no new attention extraction
[ ] no new Test inference
[ ] no training / scaler fit / optimizer
[ ] no head / seed / case selection / cherry-pick
[ ] notebook UNCHANGED
[ ] Phase54 + later NOT started
```

---

## 9. Notebook and downstream scope

```text
Notebook:            NOT modified in this run.
Phase 53-H:          DEFERRED — requires separate Human approval.
Phase 54-57:         UNAUTHORIZED — only Phase54 context handoff may be written.
Phase54 numeric:     Phase52 RAW LAST_QUERY NPZ (NOT PNG).
do_not_digitize_png: true.
```

---

## 10. Static safety scan contract

Phase 53 production path MUST NOT execute any of:

```text
torch.load
model.forward / model.forward_with_attention
return_attention
materialize_phase52
extract_attention
training / fine-tune
optimizer.step / .backward
fit / fit_transform / scaler.fit
prediction generation
new Test inference
notebook manipulation
```

A static scanner (regex-based) verifies this on the implementation files at the end of Phase 53-G.

---

## 11. Status: APPROVED

This plan is APPROVED in this prompt by virtue of the §7.37.14 amendment activation, provided the runtime implementation faithfully maps this plan and the canonical Phase 53 detail.
