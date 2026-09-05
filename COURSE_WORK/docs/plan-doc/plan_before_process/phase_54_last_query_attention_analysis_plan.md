# Phase 54 — Last-Query Attention Analysis — Pre-Process Plan

**Plan ID:** `phase_54_last_query_attention_analysis_plan`
**Plan version:** v1.0 (2026-09-04)
**Status:** APPROVED
**Phase:** 54 — Last-Query Attention Analysis
**Output version:** `LAST_QUERY_ATTENTION-v1`
**Architecture amendment:** v1.16 (2026-09-04) — `phase-54-architecture-amendment-v1.16`
**Canonical detail doc:** `docs/plan-doc/plan_detail_for_each_phase/Phase_54_Last-query_attention.md`
**Governance log:** `docs/save_log_in_processing/phase_54_architecture_amendment_log.json`

---

## 1. Scope

Phase 54 is a strictly quantitative last-query attention analysis over the frozen
Phase 52 raw last-query attention NPZ for the full FINAL_TEST_POP-v1 (N=2961)
across 3 seeds × 2 layers × 4 heads. It is an UNAUTHORIZED-for-scientific-decision
phase that produces descriptive temporal-attention characterizations only.

Numerical source is exclusively:
- `artifacts/attention_extraction/raw/last_query_attention_seed42.npz`
- `artifacts/attention_extraction/raw/last_query_attention_seed123.npz`
- `artifacts/attention_extraction/raw/last_query_attention_seed2026.npz`

PNG heatmap pixels MUST NOT be used as numeric input.

---

## 2. Sub-phase Gates

| Gate | Name | Description |
|------|------|-------------|
| 54-A | Governance + preflight + source verification | Verify Phase 52/53 signoffs, NPZ SHAs, target order, lag map, pooling, Phase 53 context handoff. Create preflight audit, source verification, target-order audit, lag-mapping audit, integrity audit. Apply architecture amendment v1.16. |
| 54-B | Contract freeze + infrastructure setup | Build `src/course_work/phase54/` package: `sources.py`, `integrity.py`, `contract.py`, `metrics.py`, `coverage.py`, `aggregations.py`, `profiles.py`, `report_cases.py`, `findings.py`, `handoffs.py`, `signoff.py`, `writers.py`, `figures.py`, `orchestrator.py`, `finalize_phase54.py`. Freeze analysis contract. |
| 54-C | Per-vector metrics + Phase 52 reconstruction audit | Compute per-target per-seed per-layer per-head metrics: entropy, normalized_entropy, effective_source_count, expected_lag_steps/minutes, lag_sd, top1 lag/weight/tie_count, top5_mass, recent_1h/6h/12h/24h_mass, Lag50/80/90. Recompute Phase 52 summary metrics from raw and verify tolerance. |
| 54-D | Aggregation: profile + per-head metric summaries | Aggregate metrics across targets: per-seed/per-layer/per-head metric summary (N, mean, sd, median, p05, p25, p75, p95). Compute mean/median/SD/p05/p25/p75/p95 temporal profiles by lag. Compute layer head-mean profiles and seed overall profiles. |
| 54-E | Lag-bin, recent-mass, coverage, top1 frequency | Compute non-overlapping lag-bin masses (1–6, 7–36, 37–72, 73–144, 145–L; only supported). Aggregate recent-mass summary. Aggregate coverage-radius summary. Aggregate top1 lag frequency and top1 tie summary. |
| 54-F | Figures + report-case line plots | Generate core figures: mean profiles per seed/layer (LASTQ_54_01..06), layer head-mean profile overlay, normalized entropy/expected-lag/top5 mass/recent-mass/non-overlap lag-bin/Lag50-80-90/top1 frequency/cumulative recency figures. Generate deterministic Phase 51 shared ranks 1–5 case-level line plots. |
| 54-G | Handoffs + findings + tests + report + signoff | Emit Phase 55 head-comparison handoff + Phase 56 error-conditioned context handoff + Phase 57 seed-stability context handoff. Write findings, discrepancies, attention tests, summary, report, README, sign-off. |

Each sub-phase requires Human approval at its gate. Phase 54-H notebook
visualization is DEFERRED and requires separate Human approval.

---

## 3. Frozen Inputs

| Artifact | Path | Purpose |
|----------|------|---------|
| Phase 52 signoff | `artifacts/attention_extraction/phase_52_signoff.json` | Verify PASS/PASS_WITH_WARNING |
| Phase 53 signoff | `artifacts/attention_heatmaps/phase_53_signoff.json` | Verify PASS + context_ready=true |
| Phase 54 numerical handoff | `artifacts/attention_extraction/phase54_last_query_attention_handoff.json` | Raw SHA256 list, target-order SHA, lag-map SHA, summary SHA, definitions |
| Phase 53 context handoff | `artifacts/attention_heatmaps/phase54_last_query_attention_context_handoff.json` | Numeric source pinned to Phase 52 raw NPZ |
| Raw last-query NPZ (3 files) | `artifacts/attention_extraction/raw/last_query_attention_seed{42,123,2026}.npz` | Primary numeric source |
| Raw checksums | `artifacts/attention_extraction/raw_attention_checksums.json` | SHA256 verification |
| Target order | `artifacts/attention_extraction/attention_test_target_order.csv` | Frozen target ordering |
| Position/lag map | `artifacts/attention_extraction/attention_relative_position_map.csv` | Frozen lag mapping (lag_steps = L - p, H=1) |
| Phase 52 last-query summary | `artifacts/attention_extraction/attention_last_query_summary.csv` | Reference for reconstruction audit |
| Phase 52 recent mass summary | `artifacts/attention_extraction/attention_recent_mass_summary.csv` | Reference for reconstruction audit |
| Phase 52 top source summary | `artifacts/attention_extraction/attention_top_source_summary.csv` | Reference for reconstruction audit |
| Phase 53 report cases | `artifacts/attention_heatmaps/attention_heatmap_report_cases.csv` | Shared ranks 1–5 for deterministic case-level line plots |
| Phase 51 case selection | `artifacts/worst_error_analysis/worst_error_analysis_case_selection.csv` | Phase 51 selection (frozen; do not re-rank) |

---

## 4. Hard Scientific Contracts (locked before any computation)

- **Numeric source:** Phase 52 float32 last-query NPZ only; no PNG digitization.
- **Last query definition:** `A[:, :, L-1, :]` over historical source positions.
- **Raw axis order:** `[target, layer, head, source]`.
- **Source position semantics:** 0 = oldest, L-1 = newest.
- **Forecast target:** NOT an attention token.
- **Lag mapping:** `lag_steps_p = L - p (H=1)`; `lag_minutes_p = 10 × lag_steps_p`.
- **Recent windows:** 1h≤6, 6h≤36, 12h≤72, 24h≤144 steps; `effective_steps = min(requested, L)`.
- **Top1 tie rule:** `NEWEST_SOURCE` among exact-tied maxima.
- **Entropy:** `H = -Σ_p a_p · log(a_p + ε_H)`; `ε_H = 1e-12`.
- **Normalized entropy:** `H / log(L)`.
- **Effective source count:** `exp(H)`.
- **Expected lag steps:** `Σ_p a_p · lag_steps_p`.
- **Lag SD steps:** `sqrt(Σ_p a_p · (lag_steps_p − E[lag])²)`.
- **Top5 mass:** sum of five largest `a_p`.
- **Lag50/80/90:** smallest `k` s.t. cumulative recency mass (newest→oldest) ≥ {0.50, 0.80, 0.90}.
- **Non-overlap lag bins:** 1–6, 7–36, 37–72, 73–144, 145–L; only supported bins emitted.
- **Mean profile audit:** `Σ_lag mean_weight ≈ 1` per seed/layer/head.
- **Layer head-mean profile:** `Σ_lag head_mean_weight ≈ 1`.
- **No head ranking, no head clustering, no head ablation, no head matching.**
- **No error-conditioned groups, no regime-conditioned groups.**
- **No cross-seed head averaging as primary; descriptive seed-level only.**
- **No feature-importance claim, no causal claim.**
- **Three seeds strictly separate; no aggregation across seeds as primary.**

---

## 5. Forbidden Actions

Phase 54 may NOT:
- Train, fine-tune, call optimizer.step, .backward, scaler.fit, scaler.fit_transform.
- Load model checkpoint, run model.forward, call return_attention, materialize_phase52.
- New attention extraction.
- New Test inference for any prediction modification.
- Best-seed selection, ensemble, seed averaging as primary result.
- Best-head selection, head ranking, head clustering, head ablation, cross-seed head matching.
- Attention labeled as feature importance or causal attribution.
- Implement Phase 55 / 56 / 57 (handoff files only).
- Destructive overwrite of any Phase 47/48/49/50/51/52/53 canonical artifact.
- Modify frozen prediction values, frozen attention values, frozen target order, frozen position map.
- Notebook modification (Phase 54-H deferred).
- Reading PNG pixels for numeric attention values.

---

## 6. Required Outputs (O54 inventory)

| ID | Filename | Required | Description |
|----|----------|----------|-------------|
| O54.1 | `last_query_attention_manifest.json` | Yes | Manifest with version, source SHAs, status |
| O54.2 | `last_query_attention_contract.json` | Yes | Frozen analysis contract |
| O54.3 | `phase54_preflight_audit.csv` | Yes | Preflight audit with critical gate status |
| O54.4 | `last_query_source_verification.csv` | Yes | Per-seed SHA / shape / dtype / count verification |
| O54.5 | `last_query_integrity_audit.csv` | Yes | Per-seed/layer/head vector integrity audit |
| O54.6 | `last_query_phase52_summary_reconstruction_audit.csv` | Yes | Recomputed Phase 52 metrics vs original |
| O54.7 | `last_query_target_order_audit.csv` | Yes | Cross-seed target-order audit |
| O54.8 | `last_query_lag_mapping_audit.csv` | Yes | Lag-mapping audit (lag1↔L-1; lagL↔0) |
| O54.9 | `last_query_metrics_long.csv` | Yes | Per-target × seed × layer × head metrics |
| O54.10 | `last_query_metric_summary_by_head.csv` | Yes | Per-seed/layer/head metric aggregate summary |
| O54.11 | `last_query_profile_by_lag.csv` | Yes | Mean/median/SD/quantile temporal profiles by lag |
| O54.12 | `last_query_layer_head_mean_profile.csv` | Yes | Layer-level head-mean profile |
| O54.13 | `last_query_seed_overall_profile.csv` | Yes | Seed-level overall descriptive profile |
| O54.14 | `last_query_lag_bin_mass.csv` | Yes | Non-overlapping lag-bin masses |
| O54.15 | `last_query_recent_mass_summary.csv` | Yes | Per-seed/layer/head recent mass summary |
| O54.16 | `last_query_coverage_radius_summary.csv` | Yes | Lag50/Lag80/Lag90 coverage summary |
| O54.17 | `last_query_top1_lag_frequency.csv` | Yes | Top1 lag frequency per seed/layer/head |
| O54.18 | `last_query_top1_tie_summary.csv` | Yes | Top1 tie summary |
| O54.19 | `last_query_report_case_manifest.csv` | Yes | Report-case manifest (Phase 51 shared ranks 1–5) |
| O54.20 | `last_query_report_case_metrics.csv` | Yes | Report-case last-query metrics |
| O54.21 | `figures/*.png` (15+ figures) | Yes | Core figures |
| O54.21b | `figures/report_cases/*.png` | Yes | Report-case line plots |
| O54.22 | `last_query_analysis_findings.csv` | Yes | Findings (descriptive only) |
| O54.23 | `phase55_head_comparison_handoff.json` | Yes | Phase 55 handoff |
| O54.24 | `phase56_error_conditioned_attention_context_handoff.json` | Yes | Phase 56 context handoff |
| O54.25 | `phase57_seed_stability_attention_context_handoff.json` | Yes | Phase 57 context handoff |
| O54.26 | `last_query_attention_tests.csv` | Yes | Test inventory |
| O54.27 | `last_query_attention_discrepancies.json` | Yes | Discrepancy log |
| O54.28 | `last_query_attention_summary.json` | Yes | Machine-readable summary |
| O54.29 | `last_query_attention_report.md` | Yes | Human-readable report (24 sections) |
| O54.30 | `README_LAST_QUERY_ATTENTION.md` | Yes | README |
| O54.31 | `phase_54_signoff.json` | Yes | Final sign-off |

---

## 7. Source Package Plan (`src/course_work/phase54/`)

```
src/course_work/phase54/
  __init__.py
  sources.py             # Frozen paths, constants, target-order/lag-map SHAs
  integrity.py           # Per-vector integrity recheck + Phase 52 reconstruction audit
  contract.py            # Freeze analysis contract + write contract artifact
  metrics.py             # Per-vector metrics: entropy, normalized_entropy, effective_source_count, expected_lag, lag_sd, top1, top5, recent
  coverage.py            # Lag50/80/90 + non-overlapping lag-bin masses
  aggregations.py        # Per-head metric summary + temporal profiles by lag
  profiles.py            # Layer head-mean + seed overall profiles
  report_cases.py        # Resolve deterministic Phase 51 shared ranks 1–5 + write report case manifest/metrics
  figures.py             # Core figure renderers + report-case line plots
  findings.py            # Findings + discrepancies writers
  handoffs.py            # Phase 55/56/57 handoff writers
  signoff.py             # Final Phase 54 sign-off writer
  writers.py             # Atomic CSV/JSON writers
  orchestrator.py        # Run all sub-phases (54-A through 54-G)
  finalize_phase54.py    # Generate report/README/summary/signoff + static safety scan
tests/unit/test_phase54_*.py
```

---

## 8. Output Directory (`artifacts/last_query_attention/`)

```
artifacts/last_query_attention/
  last_query_attention_manifest.json
  last_query_attention_contract.json
  phase54_preflight_audit.csv
  last_query_source_verification.csv
  last_query_integrity_audit.csv
  last_query_phase52_summary_reconstruction_audit.csv
  last_query_target_order_audit.csv
  last_query_lag_mapping_audit.csv
  last_query_metrics_long.csv
  last_query_metric_summary_by_head.csv
  last_query_profile_by_lag.csv
  last_query_layer_head_mean_profile.csv
  last_query_seed_overall_profile.csv
  last_query_lag_bin_mass.csv
  last_query_recent_mass_summary.csv
  last_query_coverage_radius_summary.csv
  last_query_top1_lag_frequency.csv
  last_query_top1_tie_summary.csv
  last_query_report_case_manifest.csv
  last_query_report_case_metrics.csv
  last_query_analysis_findings.csv
  last_query_attention_tests.csv
  last_query_attention_discrepancies.json
  last_query_attention_summary.json
  last_query_attention_report.md
  README_LAST_QUERY_ATTENTION.md
  phase_54_signoff.json
  phase55_head_comparison_handoff.json
  phase56_error_conditioned_attention_context_handoff.json
  phase57_seed_stability_attention_context_handoff.json
  o54_inventory.json
  figures/
    LASTQ_54_01..06_mean_profiles_seed{42,123,2026}_layer{01,02}.png
    LASTQ_54_07_layer_head_mean_profiles.png
    LASTQ_54_08_normalized_entropy_by_head.png
    LASTQ_54_09_expected_lag_by_head.png
    LASTQ_54_10_top5_mass_by_head.png
    LASTQ_54_11_recent_mass_by_head.png
    LASTQ_54_12_nonoverlap_lag_bins.png
    LASTQ_54_13_lag50_lag80_lag90.png
    LASTQ_54_14_top1_lag_frequency.png
    LASTQ_54_15_cumulative_recency_profiles.png
    report_cases/
      SHARED_R{rank:02d}_SEED{seed}_LAST_QUERY.png
```

---

## 9. Hard Gates (must PASS before Phase 54-G)

- Phase 52 signoff: PASS or PASS_WITH_WARNING.
- Phase 53 signoff: PASS.
- Phase 53 context handoff: `phase54_ready=true`, `context_ready=true`.
- Phase 52 numerical handoff: `ready_for_phase54=true`.
- Raw last-query NPZ SHA256: 3/3 match.
- Raw shape: `[2961, 2, 4, 72]` per seed.
- Raw dtype: `float32`.
- Target order: identical across 3 seeds.
- Position map: lag1↔pos L-1, lagL↔pos 0 verified.
- Pooling: `LAST_STEP` from Phase 52 handoff.
- Per-vector probability audit: finite, nonnegative within tolerance, sum ≈ 1, length = L.
- Phase 52 summary reconstruction: zero material mismatch.
- No new extraction, no model load, no training.
- Phase 55 / 56 / 57: not started.

---

## 10. Safety & Notebook Boundary

- Phase 54 notebook cells: NONE in this run.
- No `CourseWork_1.ipynb` modification.
- No Run All.
- Renderer safety scan: phase 54 source must NOT contain forbidden operations
  (`torch.load(`, `model(`, `return_attention`, `np.load(` for any non-Phase 52
  source, `materialize_phase52`, `optimizer`, `backward`, `fit`, `new inference`,
  `attention extraction`).
- Static safety scan runs at the end of Phase 54-G.

---

## 11. Acceptance Criteria

Phase 54 PASS only when:
- All Phase 52 raw last-query NPZ files verified by SHA256, shape, target ordering, lag mapping.
- Last-query definition treated as `A[:, :, L-1, :]` over historical sources.
- All vectors finite, nonnegative within tolerance, sum ≈ 1, length = L.
- Phase 52 last-query summaries independently reconstructed from raw vectors and agree under frozen tolerance.
- Raw positions mapped to lag steps using `H + (L-1-p), H=1`; recency cumulative analyses reorder newest→oldest.
- Per-target per-seed per-layer per-head metrics include all required fields.
- Recent 1h/6h/12h/24h summaries record truncation flags where `L < requested`.
- Non-overlapping lag-bin masses cover supported lookback without overlap/gap and sum ≈ 1.
- All-Test mean temporal attention profiles for every seed/layer/head; each mean profile sum ≈ 1.
- Layer head-mean profiles produced without replacing per-head profiles.
- Top1 lag frequency and tie frequency computed using NEWEST_SOURCE tie rule.
- All aggregate summaries preserve architectural head order and do not rank heads.
- Phase 51 shared ranks 1–5 used only as illustrative case-level last-query views.
- No new attention extraction, no new Test inference, no head clustering, no head ablation, no error conditioning, no seed-stability claim, no feature-importance claim, no causal attribution.
- Phase 55 receives standardized head metrics/profiles; Phase 56/57 receive context only.
