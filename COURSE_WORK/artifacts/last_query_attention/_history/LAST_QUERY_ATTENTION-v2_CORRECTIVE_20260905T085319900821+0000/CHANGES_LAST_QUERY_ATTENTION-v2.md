# Phase 54 — Corrective Changes Log (LAST_QUERY_ATTENTION-v1 → v2)

This file documents the audit-driven scientific corrective that produced
LAST_QUERY_ATTENTION-v2 active revision.

## Corrective Authorization
- Authorized after read-only Phase 54 audit found HIGH-severity `normalized_entropy`
  data-integrity defect in LAST_QUERY_ATTENTION-v1 artifacts.
- Authorization basis: Phase 54 audit report (HIGH severity triggers corrective).

## Corrective Scope
- Repair Phase 54 derived metrics, contract schema, and dependent handoffs.
- DO NOT touch raw Phase 52 last-query tensors.
- DO NOT touch predictions, residuals, checkpoints, scalers, target selection,
  regimes, head/seed rankings.

## Defect Resolved #1 — HIGH severity
- **Field**: `normalized_entropy` column in last_query_metrics_long.csv and
  last_query_metric_summary_by_head.csv
- **Symptom v1**: All 24 (seed, layer, head) groups ⇒ mean=0.0, sd=0.0, all
  quantiles = 0.0, status=PASS (falsely)
- **Root cause**: `src/course_work/phase54/metrics.py:163` called
  `normalized_entropy(a, lag_steps)` but the function signature is
  `(a, eps=EPSILON_H)`. The `lag_steps` numpy array (shape (72,)) was broadcast
  into the eps slot of `shannon_entropy`; internal `np.clip(q, eps, 1.0)` clamped
  all probabilities to 1.0; entropy became 0; `H_norm = 0 / log(1.0) = 0.0`.
- **Fix**: Changed `normalized_entropy(a, lag_steps)` → `normalized_entropy(a)` so
  the call uses the default `eps=EPSILON_H=1e-12`. Other call sites
  (notably `integrity.py:171`) already used the correct inline formula `H / np.log(LOOKBACK)`,
  which is why the Phase 52 reconstruction audit PASSed in v1.

## Defect Resolved #2 — LOW severity
- **Field**: `non_overlap_bins` schema and `lag_bin_mass.csv` BIN_BEYOND_24H rows
- **Symptom v1**: Final entry was `[145, LOOKBACK]` which for L=72 collapsed to
  `[145, 72]` — an inverted interval (start > end). BIN_BEYOND_24H rows were
  tagged `applicable=True, status=PASS` despite being mathematically unreachable.
- **Fix**:
  - `last_query_attention_contract.json`: `non_overlap_bins` final entry
    changed to `[145, 240]` (canonical upper bound) with explicit
    `non_overlap_bin_applicability_under_current_lookback = False` for L=72.
  - `last_query_lag_bin_mass.csv`: 71,064 BIN_BEYOND_24H rows patched
    `applicable=True → False`, `status=PASS → NOT_APPLICABLE`. `effective_start`
    and `effective_end` recomputed consistently to `max(raw_start, 1)` and
    `min(raw_end, LOOKBACK)` respectively.
  - BIN_GT12H_TO_24H was already flagged `applicable=False` and is unchanged.
  - Three applicable bins (BIN_10M_TO_1H, BIN_GT1H_TO_6H, BIN_GT6H_TO_12H)
    unchanged numerically (mass values preserved).

## Unaffected (regression audit, byte-level confirmed)
- `last_query_metrics_long.csv` — all 36 non-normalized_entropy columns:
  byte-identical to v1 except the normalized_entropy column.
- `last_query_metric_summary_by_head.csv` — 360/384 rows untouched (other metrics).
- `entropy` column — all 24 (seed, layer, head) groups byte-identical.
- `last_query_attention_tests.csv` — 53/54 tests unchanged; T54.E03 status updated to PASS_V2_CORRECTED.
- All raw Phase 52 NPZ files: unchanged SHA256.
- Test population order: c7039090… unchanged.
- Lag map: 948bc0dc… unchanged.
- All 24 (seed, layer, head) head summaries: `entropy`, `effective_source_count`,
  `expected_lag_steps/minutes`, `lag_sd_steps/minutes`, `top1_*`, `top5_mass`,
  `recent_*` columns byte-identical to v1.

## Verification
- Independent audit:
  `phase54_corrective_independent_normalized_entropy_audit.json`
- 24/24 (seed, layer, head) groups: stored vs independently reconstructed
  mean H_norm agrees within 2.19e-09 max abs error (tolerance 1e-6).
- 100-vector spot check (seed 42, layer 0, head 0): all within 1e-7 absolute.
- H_norm validity range: 0.000006 ≤ H_norm ≤ 0.993418 (1 over-range within 1e-9).
- No hard-coded expected values used; only the formula H / ln(72) was validated.

## Artifacts Updated (v2 active revision)
- `last_query_metric_summary_by_head.csv` — 24 rows patched
- `last_query_metrics_long.csv` — 71,064 rows patched
- `last_query_lag_bin_mass.csv` — 71,064 BIN_BEYOND_24H rows patched
- `last_query_attention_contract.json` — schema/bin corrections
- `last_query_attention_manifest.json` — version bumped
- `last_query_attention_summary.json` — corrective metadata
- `last_query_attention_discrepancies.json` — 2 items resolved (D54.CORR.*)
- `o54_inventory.json` — version bumped
- `last_query_attention_tests.csv` — T54.E03 status updated
- `phase_54_signoff.json` — version bumped
- `phase55_head_comparison_handoff.json` — source_phase54_version=v2, artifact SHAs updated
- `phase56_error_conditioned_attention_context_handoff.json` — same
- `phase57_seed_stability_attention_context_handoff.json` — same

## Source Code Updates
- `src/course_work/phase54/metrics.py:163` — call signature corrected
  (`normalized_entropy(a, lag_steps)` → `normalized_entropy(a)`).
- `src/course_work/phase54/sources.py:54-94` — `NON_OVERLAP_BINS` final entry fixed
  (`(145, LOOKBACK)` → `(145, 240)`); legacy declaration preserved as
  `NON_OVERLAP_BINS_RAW_PLAN` for traceability; new
  `NON_OVERLAP_BIN_APPLICABILITY_UNDER_LOOKBACK` dict records per-bin applicability.

## Notebooks
- `notebook_course_work/CourseWork_1.ipynb` Phase 54 dashboard cell (139):
  currently renders LAST_QUERY_ATTENTION-v1 cached content.
  NOT MODIFIED in this corrective (presentation-only repair step is
  separate and requires its own approval).
- A future presentation refresh will pick up the v2 corrected CSV by simply
  re-running the dashboard cell (read-only).

## Downstream Phase Authorization State
- `phase55_authorized = false` (unchanged)
- `phase56_authorized = false` (unchanged)
- `phase57_authorized = false` (unchanged)
- Downstream Phase 55/56/57 may now consume v2 corrected artifacts, but their
  IMPLEMENTATION remains BLOCKED pending separate Human gates.

## v1 Archive
- The original LAST_QUERY_ATTENTION-v1 active directory has been archived to:
  `artifacts/last_query_attention/_history/LAST_QUERY_ATTENTION-v1_ARCHIVED_<ts>/`
- Archive SHA-256 manifest: `_archive_manifest.json` inside that directory.

## Source-of-Truth Frozen Lineage (UNCHANGED)
- raw Phase 52 last-query NPZ (unchanged, frozen):
  - seed 42: 102086f71ed01611b963c44926d7472a3ecc49a0b63f41d79100ef816b52a9ff
  - seed 123: 00e2f8e7852908764b3712d9b172faa644dc69d176241962bc14c68fee31110e
  - seed 2026: 34051966fc47bf96f4e288ac5e984815feff217ad65828882383ee4643cc5a19
- Test population SHA: c7039090dc4d8168b24de04ebc671728f539fe873e02f1bb87da56e403f91706
- Position/lag map SHA: 948bc0dcb56674f7f8c117d7e7050490b391e39225754e8a21861d00e59e5063
