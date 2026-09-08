# Phase 50 — Error-by-Regime Analysis (Pre-Process Plan)

**Project:** UCI Appliances Energy Prediction
**Task:** Multivariate Time-Series Regression
**Forecasting contract:** Sequence-to-One, One-Step-Ahead
**Residual convention:** `residual = y_true - y_pred`
**Phase:** 50 — ERROR-BY-REGIME ANALYSIS
**Phase version (target):** `ERROR_BY_REGIME-v1`
**Phase detail:** `docs/plan-doc/plan_detail_for_each_phase/Phase_50_Error-by-regime_analysis.md`
**Architecture file:** `docs/RULE_BASE/architecture_rule.md`
**Rule code file:** `docs/RULE_BASE/rule_code.md`
**This plan version:** v1.0
**Status:** DRAFT — AWAITING human approval (per rule_code.md §10–§17) AND architecture amendment v1.12 approval
**Created at:** 2026-09-04

---

## 1. Objective

Phase 50 performs a strictly **diagnostic** (not generative) analysis of Phase 47 prediction error on the held-out Test (N_TEST = 2961) by partitioning Test target_ids into **six pre-declared regime families** whose thresholds are derived **exclusively from the original Train** (REGIME_REFERENCE_TRAIN-v1). It answers whether the three frozen Transformer prediction bundles (seeds 42 / 123 / 2026) over- or under-error in specific regimes (target level, extreme high demand, change magnitude, change direction, time of day, day type) — without any Test-derived threshold, any retuning, any best-seed selection, any ensemble promotion, any post-hoc correction, any worst-error ranking, and any attention analysis.

The **goal is regime-attributable diagnostic classification**, not model retuning and not regime-specific correction.

---

## 2. Upstream frozen state (verified)

| Upstream | Required artifact | Required status | Verified at audit time |
|---|---|---|---|
| Phase 47 Final Test Evaluation | `artifacts/final_test/predictions/final_test_predictions_{seed42,seed123,seed2026,persistence}.csv` | byte-identical to `prediction_checksums.json` | **PASS** — 4/4 sha256 verified |
| Phase 47 Final Test Evaluation | `artifacts/final_test/phase_47_signoff.json` | `overall_status == PASS`, `ready_for_phase48 == true` | **PASS** |
| Phase 47 Final Test Evaluation | `artifacts/final_test/final_test_population_manifest.json` | `target_ids_sha256 == d7dbc0b3cde772dc3f62c181f0a09a4a7a5b0e7c78e567361dbfde089578da87`, `target_count == 2961` | **PASS** |
| Phase 47 Final Test Evaluation | `artifacts/final_test/prediction_checksums.json` | all 4 prediction sha256 entries match filesystem | **PASS** |
| Phase 48 Prediction Analysis | `artifacts/prediction_analysis/phase_48_signoff.json` | `overall_status == PASS`, `phase48_d4_closed == true`, `ready_for_phase49 == true` | **PASS** |
| Phase 48 Prediction Analysis | `artifacts/prediction_analysis/prediction_seed_spread.csv` | per-target seed-spread source for O50.26 | **PASS** (487520 bytes, target_id-level granularity) |
| Phase 48 Prediction Analysis | `artifacts/prediction_analysis/prediction_top_seed_disagreement.csv` | 20-row per-target ranking (NOT worst-error) | **PASS** |
| Phase 49 Residual Analysis | `artifacts/residual_analysis/phase_49_signoff.json` | `status == PASS`, `n_gates_passed == 38/38`, `n_o49_complete == 33/33`, `ready_for_phase50 == true` | **PASS** |
| Phase 49 Residual Analysis | `artifacts/residual_analysis/residual_long_table.csv` | 8883 rows (3 seeds × 2961), residual=y_true-y_pred, sign column present | **PASS** |
| Phase 49 Residual Analysis | `artifacts/residual_analysis/residual_wide_table.csv` | 2961 rows, per-seed columns + seed_mean_residual | **PASS** |
| Phase 49 Residual Analysis | `artifacts/residual_analysis/phase49_signed_bias.csv` | per-seed signed-bias summary | **PASS** |
| Phase 49 Residual Analysis | `artifacts/residual_analysis/phase49_cross_seed_sign_consensus.csv` | 6-class taxonomy (ALL_UNDER/ALL_OVER/ALL_EXACT/TWO_UNDER_ONE_OVER/TWO_OVER_ONE_UNDER/MIXED) | **PASS** |
| Phase 49 Residual Analysis | `artifacts/residual_analysis/phase50_handoff.json` | `phase50_target_regime_policy = TRAIN_DERIVED_ONLY`, `residual_convention = "y_true - y_pred"` | **PASS** |
| Phase 49 Residual Analysis | `artifacts/residual_analysis/phase51_context_handoff.json` | confirms Phase 51 first phase allowed to rank worst errors | **PASS** |
| Architecture | `docs/RULE_BASE/architecture_rule.md` v1.11 | explicit Phase 50 authorization | **NOT YET** — v1.11 explicitly forbids Phase 50 implementation; new amendment v1.12 REQUIRED (proposal in `docs/save_log_in_processing/phase_50_architecture_amendment_log.json`) |
| SPLIT-v1 | `artifacts/splits/split_manifest.json` | `split_version == SPLIT-v1`, `split_method == chronological_observation_proportion`, `train_rows == 13814` | **PASS** |
| WINDOWPOP-v1 | `artifacts/windows/window_manifest.json` + `window_fingerprints.json` | `population_version == WINDOWPOP-v1`, `common_population_fingerprint == a40ded88…1987`, `primary_boundary_protocol == WB0_CONTEXT_CARRY_OVER`, `common_target_population_policy == intersection_of_native_L36_L72_L144_WB0_targets` | **PASS** |
| WINDOWPOP-v1 (Train subset) | `artifacts/windows/common_target_population.csv` (filtered `target_split_id == TRAIN`) | 13670 Train target_ids; needed for REGIME_REFERENCE_TRAIN-v1 reconstruction | **PASS** |
| TEMPORAL-v1 | `artifacts/temporal/temporal_manifest.json` | canonical cadence = 10 min, continuity_segment_id column | **PASS** |
| FEATURESETS-v1 | `artifacts/features/feature_engineering_manifest.json` | raw lineage preserved, calendar features on TRAIN rows only | **PASS** |
| Raw Appliances Wh | `data/raw_data/energydata_complete.csv` (19735 rows, 0 NaN) | train target_id → raw_row index 0-based = timeline_target − 1 | **PASS** |
| LSTM eligibility | `artifacts/final_test/final_test_lstm_eligibility.json` | `eligibility_status == NOT_ELIGIBLE_CONFIG_MISMATCH` | **PASS** — Phase 50 will emit a single NOT_APPLICABLE row for LSTM |
| Persistence baseline | `artifacts/final_test/predictions/final_test_predictions_persistence.csv` (sha256=7115af1c…, N=2961) | on same Test population, target_id-level | **PASS** |

---

## 3. Downstream consumers

| Downstream | Required artifact from Phase 50 | Hard constraint |
|---|---|---|
| Phase 51 Worst-Error Analysis | `phase50_regime_worst_error_context.json` (NEW) | Test target_ids × regime_label × per-seed abs/sq error, NO top-K ranking (Phase 51 owns that) |
| Phase 52+ Attention Analysis | `phase50_regime_attention_context.json` (NEW) | regime_label + target_id list per regime; NO attention extraction |

---

## 4. Files to read (sequential — execution order, but Phase 50-A READS ONLY)

1. `docs/RULE_BASE/working_rule.md` (mandatory first read per rule_code §42)
2. `docs/RULE_BASE/architecture_rule.md` v1.11 (confirmed Phase 50 NOT authorized; v1.12 proposal created)
3. `docs/RULE_BASE/rule_code.md` (already read)
4. `docs/plan-doc/plan_detail_for_each_phase/Phase_50_Error-by-regime_analysis.md` (FULL read; 4380 lines)
5. `docs/plan-doc/plan_detail_for_each_phase/Phase_49_Residual_analysis.md` (for upstream convention)
6. `docs/plan-doc/plan_before_process/phase_50_error_by_regime_analysis_plan.md` (this file)
7. `docs/plan-doc/plan_before_process/phase_49_residual_analysis_plan.md` (style reference)
8. `artifacts/splits/split_manifest.json` + `split_summary.csv` + `split_membership.csv`
9. `artifacts/windows/window_manifest.json` + `window_fingerprints.json` + `common_target_population.csv`
10. `artifacts/temporal/temporal_manifest.json` + `temporal_summary.csv`
11. `artifacts/features/feature_engineering_manifest.json` + `feature_engineering_audit.csv`
12. `artifacts/final_test/phase_47_signoff.json` + `final_test_summary.json` + `final_test_population_manifest.json` + `prediction_checksums.json` + `final_test_lstm_eligibility.json`
13. `artifacts/prediction_analysis/phase_48_signoff.json` + `prediction_seed_spread.csv` + `prediction_top_seed_disagreement.csv`
14. `artifacts/residual_analysis/phase_49_signoff.json` + `phase50_handoff.json` + `phase51_context_handoff.json` + `residual_long_table.csv` + `residual_wide_table.csv` + `phase49_signed_bias.csv` + `phase49_cross_seed_sign_consensus.csv`
15. `data/raw_data/energydata_complete.csv` (raw Appliances Wh source for REGIME_REFERENCE_TRAIN-v1 reconstruction)
16. `tests/unit/test_phase49_f_finalization.py` (style reference for Phase 50 unit tests)

---

## 5. Filed-of-truth mapping (Phase 50 plan §3 vs actual canonical filenames)

| Phase 50 plan logical requirement | Actual canonical artifact | Status | Action |
|---|---|---|---|
| `phase_49_signoff.json` | `artifacts/residual_analysis/phase_49_signoff.json` | EXACT_MATCH | none |
| `phase50_error_regime_handoff.json` | `artifacts/residual_analysis/phase50_handoff.json` (TRAIN_DERIVED_ONLY policy handoff) AND `artifacts/final_test/phase50_error_regime_handoff.json` (541-byte placeholder from Phase 47) | AMBIGUOUS | Phase 50 implementation reads `artifacts/residual_analysis/phase50_handoff.json` as the consumption handoff; OVERWRITES `artifacts/final_test/phase50_error_regime_handoff.json` with a new `phase50_signoff_input.json` documenting Phase 50's input gates (do NOT create duplicate names). |
| `residual_long_table.csv` | `artifacts/residual_analysis/residual_long_table.csv` | EXACT_MATCH | none |
| `residual_wide_table.csv` | `artifacts/residual_analysis/residual_wide_table.csv` | EXACT_MATCH | none |
| `residual_bias_summary.csv` | `artifacts/residual_analysis/phase49_signed_bias.csv` | CANONICAL_EQUIVALENT | Phase 50 reads `phase49_signed_bias.csv`; do NOT create `residual_bias_summary.csv` alias. |
| `residual_acf_diagnostics.csv` | `artifacts/residual_analysis/phase49_residual_acf.csv` (full) + `phase49_residual_acf_key_lags.csv` (key lags) | PARTIAL_CONFLICT | Phase 50 reads `phase49_residual_acf_key_lags.csv` for narrative reference only; does NOT create new ACF artifacts (Phase 49 owns ACF). |
| `residual_seed_sign_consensus.csv` | `artifacts/residual_analysis/phase49_cross_seed_sign_consensus.csv` (6-class taxonomy) | CANONICAL_EQUIVALENT | Phase 50 joins this taxonomy (ALL_UNDER / ALL_OVER / ALL_EXACT / TWO_UNDER_ONE_OVER / TWO_OVER_ONE_UNDER / MIXED) with `test_regime_assignment.csv` to produce O50.27; do NOT create a new `residual_seed_sign_consensus.csv`. Plan §117 abbreviation to 3 classes is descriptive shorthand, not a redefinition. |
| `phase_47_signoff.json` | `artifacts/final_test/phase_47_signoff.json` | EXACT_MATCH | none |
| `final_test_population_manifest.json` | `artifacts/final_test/final_test_population_manifest.json` | EXACT_MATCH | none |
| `prediction_checksums.json` | `artifacts/final_test/prediction_checksums.json` | EXACT_MATCH | none |
| SPLIT-v1 contract | `artifacts/splits/split_manifest.json` (version=SPLIT-v1) | EXACT_MATCH | none |
| WINDOWPOP-v1 contract | `artifacts/windows/window_manifest.json` (population_version=WINDOWPOP-v1) + `window_fingerprints.json` | EXACT_MATCH | none |
| TEMPORAL-v1 contract | `artifacts/temporal/temporal_manifest.json` (cadence_minutes=10) | EXACT_MATCH | none |
| FEATURESETS-v1 contract | `artifacts/features/feature_engineering_manifest.json` | EXACT_MATCH | none |
| `final_test_predictions_persistence.csv` | `artifacts/final_test/predictions/final_test_predictions_persistence.csv` (sha256=7115af1c…, N=2961) | EXACT_MATCH | none |

**Do NOT create duplicate compatibility files** for any of the above. Phase 50 always reads from the canonical location.

---

## 6. Train reference construction (REGIME_REFERENCE_TRAIN-v1)

**Conceptual definition** (per Phase 50 plan §4):

```
REGIME_REFERENCE_TRAIN-v1
  =  original Train target_ids
   ∩  WINDOWPOP-v1 target_ids
   ∩  continuity-valid target_ids (per common_target_population.csv)
   ∩  temporally valid target_ids (cadence = 10 min, no gap)
   ∩  raw_target_coordinate == "Appliances Wh from data/raw_data/energydata_complete.csv"
```

**Concrete construction recipe** (to be executed in Phase 50-B):

1. Load `artifacts/splits/split_manifest.json` → fix SPLIT-v1 fingerprint `4d0115b92a3c81406b62deb4d36dd8ceac847f948294cdc2e4dc49338ebeb821`.
2. Load `artifacts/windows/common_target_population.csv` → filter `target_split_id == "TRAIN"` (expected N = 13,670).
3. Cross-join with `artifacts/temporal/temporal_manifest.json` to confirm cadence_minutes=10.
4. Cross-join with `data/raw_data/energydata_complete.csv` using mapping `raw_row_index = timeline_target − 1` (timeline_target is 1-based; raw_row_index is 0-based) to retrieve the raw `Appliances` Wh for each Train target_id.
5. Compute `REGIME_REFERENCE_TRAIN-v1` SHA256 over the sorted list of `target_sample_id` strings (UTF-8, LF newline-separated); expect a deterministic 64-hex output.
6. Emit `artifacts/error_by_regime/regime_reference_train_manifest.json` with the sha256, N, and provenance chain (SPLIT-v1 fp, WINDOWPOP-v1 fp, raw_data_source_path).
7. **Reject any target_id** that lacks a valid raw Appliances Wh or has a non-10-min gap to its predecessor.

**EXCLUDED populations** (verbatim from Phase 50 plan §4):

- VALIDATION (2960 targets; timeline 13814..16773) — used for early-stopping only, never for threshold derivation.
- TEST (2961 targets; timeline 16774..19734) — only used for regime assignment and metric evaluation, never for threshold derivation.
- Train+Validation final-refit population — train+val retrained Transformer (if any) is NOT a Train reference; per Phase 50 plan §4.
- YS1 standardized `y_model` — Phase 50 plan §6 explicitly forbids standardized coordinates.
- RevIN normalized coordinates — Phase 50 plan §6 explicitly forbids.

---

## 7. Threshold derivation contract (Stage A — TRAIN ONLY)

| Threshold | Family | Source variable | Method | Expected frozen value (Phase 50-B will compute & freeze) |
|---|---|---|---|---|
| Q25_y | R1 target_level | raw Appliances Wh of REGIME_REFERENCE_TRAIN-v1 | numpy.quantile @ p=0.25, method=LINEAR (numpy default) | TBD in Phase 50-B |
| Q75_y | R1 target_level | raw Appliances Wh of REGIME_REFERENCE_TRAIN-v1 | numpy.quantile @ p=0.75, method=LINEAR | TBD |
| Q90_y | R2 extreme_high | raw Appliances Wh of REGIME_REFERENCE_TRAIN-v1 | numpy.quantile @ p=0.90, method=LINEAR | TBD |
| Q90_abs_delta | R3 change_magnitude | abs(Δy) over REGIME_REFERENCE_TRAIN-v1 pairs where predecessor is exactly 10 min earlier AND in the same continuity segment AND no split crossing | numpy.quantile @ p=0.90, method=LINEAR | TBD |

**Library / function** (frozen):
- `numpy.quantile(values, q, method="linear")` — numpy default for 1-D array.
- Input dtype: `numpy.float64`. Non-finite values (NaN, inf) rejected before quantile computation.
- Deterministic rerun: same Train reference + same numpy version + same method = bit-identical thresholds.

**Degeneracy handling** (per Phase 50 plan §13):
- If `Q25_y >= Q75_y` after Train-only computation → mark `TARGET_LEVEL_THRESHOLD_DEGENERATE` and STOP R1 execution pending human-approved resolution. Do NOT perturb thresholds using Test.
- If `Q90_abs_delta ≤ 0` (no positive deltas) → mark `CHANGE_MAGNITUDE_THRESHOLD_DEGENERATE` and STOP R3 execution.

**Fingerprint** (frozen before any Test assignment):
- `regime_threshold_fingerprint.json` records: sha256(`regime_thresholds_train_only.json`), Train target_ids_sha256, quantile_method, source_code_version_optional, status.

---

## 8. Temporal / delta contract (gap-safe, no split crossing, exact 10-min)

**Δy definition** (locked, per Phase 50 plan §17, §19, §20):

```
Δy_t  =  y_t  −  y_(t-1)
        IF AND ONLY IF:
            (a) y_(t-1) exists in same continuity_segment_id (per common_target_population.csv)
            AND (b) target_timestamp(t) − target_timestamp(t-1) == 10 minutes (exact)
            AND (c) t-1 is on the same split side of WINDOWPOP-v1 (no split crossing)
```

**WB0 Test predecessor authorization** (per Phase 50 plan §20):
- The first Test target (TGT_00016774, timeline_target=16774, 2016-05-07 04:40:00) may use its preceding actual observation (TGT_00016773, timeline_target=16773, 2016-05-07 04:30:00, last VALIDATION target, raw Appliances=40) IF and only if the predecessor passes conditions (a)/(b)/(c).
- Justification: SPLIT-v1 context_policy = `input_rows_may_precede_target_period_under_wb0`; Phase 50 plan §20 explicitly authorizes this.
- Subsequent Test points (TGT_00016775, TGT_00016776, …) use the immediately preceding Test point (both within FINAL_TEST_POP-v1).
- If the predecessor is unavailable (gap, split crossing) → regime = `CHANGE_UNCLASSIFIED` for R3/R4; do NOT fabricate Δy=0.

**Forbidden**:
- Crossing a temporal gap.
- Crossing the Train/Validation/Test boundary for TRAIN threshold estimation (Train deltas are computed only within the Train split).
- Using standardized or RevIN-normalized coordinates.
- Padding with Δy=0 for unclassified rows.

---

## 9. Six regime family definitions (locked, per Phase 50 plan §7, §10–§25)

| Family | Thresholds | Labels | Assignment rule | Special cases |
|---|---|---|---|---|
| R1_TARGET_LEVEL | Train Q25_y, Q75_y | TL_LOW, TL_MID, TL_HIGH | TL_LOW: y < Q25; TL_MID: Q25 ≤ y < Q75; TL_HIGH: y ≥ Q75 | Every Test target assigned exactly once; degenerate handling per §13 |
| R2_EXTREME_HIGH | Train Q90_y | EXTREME_HIGH, NON_EXTREME | EH: y ≥ Q90; NON_EH: y < Q90 | R2 is NOT a replacement for Phase 51 (Phase 50 plan §16) |
| R3_CHANGE_MAGNITUDE | Train Q90_abs_delta | CHANGE_NORMAL, CHANGE_RAPID, CHANGE_UNCLASSIFIED | RAPID: |Δy| ≥ Q90_abs_delta; NORMAL: |Δy| < Q90_abs_delta; UNCLASSIFIED: no predecessor | Every Test target classified or explicitly UNCLASSIFIED |
| R4_CHANGE_DIRECTION | (none — sign-based) | DIR_UP, DIR_FLAT, DIR_DOWN, DIR_UNCLASSIFIED | UP: Δy > 0; FLAT: Δy == 0; DOWN: Δy < 0; UNCLASSIFIED: no predecessor | NO epsilon; exact equality only |
| R5_TIME_OF_DAY | (none — clock-based) | TOD_NIGHT, TOD_MORNING, TOD_AFTERNOON, TOD_EVENING | NIGHT 00:00-05:59; MORNING 06:00-11:59; AFTERNOON 12:00-17:59; EVENING 18:00-23:59 | Uses raw dataset timestamp clock (not UTC); `weekend` flag already in feature_engineered_v1.csv |
| R6_DAY_TYPE | (none — calendar-based) | DAY_WEEKDAY, DAY_WEEKEND | WEEKDAY Mon-Fri; WEEKEND Sat-Sun | No holiday regime (per Phase 50 plan §25) |

---

## 10. Stage A/B/C/D leakage boundary (HARD; per Phase 50 plan §28–§31)

```
Stage A  (Train-only, NO Test access):
   REGIME_REFERENCE_TRAIN-v1 → thresholds → regime_thresholds_train_only.json → regime_threshold_fingerprint.json
   Allowed inputs:  data/raw_data/energydata_complete.csv, artifacts/windows/common_target_population.csv, artifacts/splits/split_membership.csv
   Forbidden inputs: artifacts/final_test/predictions/*, artifacts/residual_analysis/{residual_long_table,residual_wide_table}.csv, artifacts/prediction_analysis/prediction_seed_spread.csv

Stage B  (Test truth + timestamp + continuity ONLY, NO predictions, NO residuals):
   test_regime_assignment.csv (NO y_pred, NO residual, NO model ID)
   Allowed inputs:  FINAL_TEST_POP-v1 target_ids + target_timestamps + continuity_segment_id + raw Appliances Wh (from y_true_wh column of seed42/123/2026 prediction bundles — same value across seeds per Phase 47 verification)
   Forbidden inputs: artifacts/residual_analysis/residual_long_table.csv, artifacts/residual_analysis/residual_wide_table.csv, y_pred_wh, residual_wh, absolute_error_wh, squared_error_wh2

Stage C  (FREEZE test_regime_assignment.csv, audit, fingerprint):
   test_regime_assignment_audit.csv (10 checks per Phase 50 plan §131)
   test_regime_assignment_fingerprint.json (sha256 of frozen assignment CSV)
   NO additional inputs allowed in this stage; NO read of any Phase 47 prediction error column

Stage D  (ONLY after Stage C, join + metric):
   regime_error_join_audit.csv (many-to-one on target_id; 0 unmatched, 0 duplicates)
   regime_metrics_long.csv (per-seed per-regime per-family)
   regime_metrics_<family>.csv (6 family-specific tables)
   Allowed inputs:  test_regime_assignment.csv (FROZEN), artifacts/residual_analysis/residual_long_table.csv, artifacts/residual_analysis/residual_wide_table.csv
   Forbidden inputs: any new inference, any scaler, any checkpoint, any retraining, any new label
```

**Module-level enforcement** (each module's import whitelist):

| Module | May import | May NOT import |
|---|---|---|
| `phase50.thresholds` | `numpy`, `pandas`, `phase50.contract`, `course_work.utils.artifacts` | `phase50.regime_assignment`, `phase50.error_join`, `phase50.metrics`, `phase49.residual` |
| `phase50.regime_assignment` | `numpy`, `pandas`, `phase50.contract`, `course_work.utils.artifacts` | `phase50.error_join`, `phase50.metrics`, `phase50.thresholds` (uses frozen output only), `phase49.residual`, `phase47.predictions` |
| `phase50.error_join` | `pandas`, `phase50.contract` | `phase50.thresholds`, `phase50.metrics`, any model / scaler / checkpoint |
| `phase50.metrics` | `numpy`, `pandas`, `phase50.contract` | `torch`, `tensorflow`, `joblib`, any checkpoint, any scaler |
| `phase50.cross_seed` | `numpy`, `pandas`, `phase50.contract`, `phase50.metrics` | `phase50.thresholds` (read-only via `regime_thresholds_train_only.json` if needed for context) |

This enforces the leakage boundary by Python import structure — making accidental Test-residual access from Stage A literally impossible at compile time.

---

## 11. Test assignment immutability

Once `test_regime_assignment.csv` is written (Stage B), it MUST be frozen:

- File SHA256 recorded in `test_regime_assignment_fingerprint.json`.
- File written atomically (tmp → rename), with mode 0444 after write (or equivalent read-only enforcement).
- Any subsequent code path that attempts to modify this file MUST fail the audit (assertion in `phase50.assignment_audit.assert_test_assignment_immutable`).
- Stage D reads the file but does not re-write it.

The audit (`test_regime_assignment_audit.csv`) MUST verify (10 checks, per Phase 50 plan §131):

1. Same target count as FINAL_TEST_POP-v1 (2961).
2. Unique target_ids.
3. Chronological order (target_timestamp ascending).
4. All 6 family label columns present (R1..R6).
5. R1 exactly one label per row (TL_LOW / TL_MID / TL_HIGH).
6. R2 exactly one label per row (EXTREME_HIGH / NON_EXTREME).
7. R5 exactly one label per row (TOD_NIGHT / TOD_MORNING / TOD_AFTERNOON / TOD_EVENING).
8. R6 exactly one label per row (DAY_WEEKDAY / DAY_WEEKEND).
9. R3 and R4 each row is one of the defined labels OR explicit `CHANGE_UNCLASSIFIED` / `DIR_UNCLASSIFIED`.
10. NO prediction columns (y_pred_wh, residual_wh, absolute_error_wh, squared_error_wh2) and NO model_id / seed columns present.

---

## 12. Per-seed metric definitions (Stage D, per Phase 50 plan §132–§138)

For each (regime_family, regime_label, model_id, seed), with N = #target_ids in the (regime_family, regime_label) × seed assignment intersection:

| Metric | Definition | Phase 50 contract |
|---|---|---|
| N | count | int |
| sample_share | N / global_N_for_seed | float, sum_to_one_per_family_per_seed |
| target_mean_wh | mean(y_true_wh) over the regime+seed subset | float |
| target_std_wh | sample std (ddof=1) | float |
| prediction_mean_wh | mean(y_pred_wh) | float |
| mae_wh | mean(|y_true_wh − y_pred_wh|) | float; MATCHES Phase 47 per-seed MAE to ≤1e-9 tolerance when summed over all regimes (audit) |
| rmse_wh | sqrt(mean((y_true_wh − y_pred_wh)²)) | float; MATCHES Phase 47 per-seed RMSE |
| r2 | 1 − Σ(y_true − y_pred)² / Σ(y_true − ȳ)² | float OR `r2_status=NOT_DEFINED` (explicit string) when denom = 0 |
| mbe_wh | mean(y_true_wh − y_pred_wh) | float (residual convention) |
| underprediction_fraction | count(residual > 0) / N | float |
| overprediction_fraction | count(residual < 0) / N | float |
| zero_fraction | count(residual == 0) / N | float |
| sae | sum(|residual|) | float |
| sse | sum(residual²) | float |
| sae_share | sae / global_sae_for_seed | float, sum_to_one_per_family_per_seed |
| sse_share | sse / global_sse_for_seed | float, sum_to_one_per_family_per_seed |
| sae_disproportion | sae_share / sample_share | float |
| sse_disproportion | sse_share / sample_share | float |
| global_mae_wh | per-seed global MAE (constant across regimes) | float |
| global_rmse_wh | per-seed global RMSE | float |
| mae_lift_pct | 100 × (mae_wh − global_mae_wh) / global_mae_wh | float |
| rmse_lift_pct | 100 × (rmse_wh − global_rmse_wh) / global_rmse_wh | float |
| rmse_lift_wh | rmse_wh − global_rmse_wh | float (SIGNED, in Wh; per Phase 50 plan §138) |
| rmse_lift_ratio | (rmse_wh / global_rmse_wh) − 1 | float (per Phase 50 plan §89 RMSELift_{s,r}; renamed from ratio to ratio to avoid confusion with rmse_lift_pct) |
| small_n_warning | True if N < 30 (arbitrary but predeclared threshold) | bool |
| status | PASS / WARN_SMALL_N / FAIL | string |

**Forbidden metric operations**:
- Best-seed selection by any metric.
- Ensemble (mean / median) across seeds as a canonical metric.
- 3N iid pooling of residuals across seeds.
- Regime-specific residual correction.

---

## 13. Contribution reconstruction audits (per Phase 50 plan §137)

For each seed:
- `Σ_{regime_label ∈ family} sample_share ≈ 1` (within 1e-9; families cover all Test target_ids).
- `Σ_{regime_label ∈ family} sae_share ≈ 1` (within 1e-9; SAE conservation).
- `Σ_{regime_label ∈ family} sse_share ≈ 1` (within 1e-9; SSE conservation).
- `Σ_{regime_label ∈ family} sse ≈ global_sse_for_seed` (within 1e-9; global SSE reconstruction).
- `Σ_{regime_label ∈ family} sae ≈ global_sae_for_seed` (within 1e-9; global SAE reconstruction).

---

## 14. Cross-seed aggregation (Stage D, per Phase 50 plan §92–§101)

For each (regime_family, regime_label, model_id=TRANSFORMER), across the 3 seeds:

| Statistic | Definition | ddof |
|---|---|---|
| mean_mae_wh | mean(per-seed mae_wh) | n/a |
| sd_mae_wh | sample std(per-seed mae_wh) | 1 |
| mean_rmse_wh | mean(per-seed rmse_wh) | n/a |
| sd_rmse_wh | sample std(per-seed rmse_wh) | 1 |
| mean_r2 | mean(per-seed r2) | n/a (only if all 3 seeds have r2_status=DEFINED; else mark r2_status=NOT_DEFINED) |
| mean_sae_share | mean(per-seed sae_share) | n/a |
| mean_sse_share | mean(per-seed sse_share) | n/a |
| mean_rmse_lift_pct | mean(per-seed rmse_lift_pct) | n/a |
| sd_rmse_lift_pct | sample std(per-seed rmse_lift_pct) | 1 |

**Strictly forbidden**:
- 3N iid pooling (`pd.concat` of all 3 seed residual rows treated as 8883 iid samples for a global regime metric).
- Best-seed selection by any cross-seed aggregate.
- Ensemble of seed predictions as a 4th model.

---

## 15. Persistence handling

Per Phase 50 plan §101–§110:

- Read `artifacts/final_test/predictions/final_test_predictions_persistence.csv` (sha256=7115af1c…, N=2961).
- For each (regime_family, regime_label), compute Persistence regime metrics using the SAME `test_regime_assignment.csv` (Test truth + timestamp + continuity only).
- Emit `regime_metrics_persistence.csv` with the same schema as `regime_metrics_long.csv` but `model_id=PERSISTENCE`.
- Compute `regime_persistence_delta_*.csv`:
  - `mae_delta_persistence_minus_transformer = mae_wh_persistence − mae_wh_transformer_seed_mean` (signed).
  - `rmse_delta_persistence_minus_transformer` (signed).
  - No sign assertion on the delta; report the direction in the dashboard.

---

## 16. LSTM eligibility handling (per Phase 50 plan §111)

`final_test_lstm_eligibility.json.eligibility_status == NOT_ELIGIBLE_CONFIG_MISMATCH` is the canonical state. Phase 50:

- Emits a single row in `regime_metrics_lstm.csv`:
  - `model_id = LSTM_TUNED_DEV`
  - `N = 0`
  - `mae_wh = ""` (empty CSV cell)
  - `rmse_wh = ""`
  - `r2_status = NOT_APPLICABLE`
  - `status = NOT_EVALUATED_BY_PROTOCOL`
- Does NOT fabricate LSTM residuals.
- Documents this in O50.29 with a `phase_50_lstm_status_note.md` style README entry.

---

## 17. Seed-spread and sign-consensus consumers

**Phase 48 → Phase 50**:
- `artifacts/prediction_analysis/prediction_seed_spread.csv` (per-target, target_id granularity) is directly joinable with `test_regime_assignment.csv` on target_id. Used for O50.26 `regime_seed_spread_summary.csv`.

**Phase 49 → Phase 50**:
- `artifacts/residual_analysis/phase49_cross_seed_sign_consensus.csv` (6-class taxonomy at target_id granularity? — actually at corpus level: `consensus_class, count, fraction`). For O50.27 `regime_seed_sign_consensus_summary.csv`, Phase 50 MUST:
  - For each target_id in the Test population, compute its per-seed residual sign (read from `residual_long_table.csv`).
  - Assign a per-target consensus class using the SAME 6-class taxonomy: ALL_UNDER (all 3 seeds have positive residual), ALL_OVER (all 3 negative), ALL_EXACT (all 3 zero), TWO_UNDER_ONE_OVER (2 positive, 1 negative), TWO_OVER_ONE_UNDER (2 negative, 1 positive), MIXED (other combinations).
  - Join with `test_regime_assignment.csv` on target_id.
  - Group by (regime_family, regime_label) and compute fraction of each consensus_class.

**Phase 49 prediction deciles**:
- `artifacts/residual_analysis/phase49_prediction_deciles.csv` is explicitly NOT a Phase 50 regime (Phase 49 lock: decile_not_phase50_regime=true). Phase 50 MUST NOT reuse decile boundaries as R1/R2/R3.
- `phase50.contract.assert_decile_not_phase50_regime(decision_made)` enforces this.

---

## 18. Figure / report plan (Phase 50-G)

Phase 50 figures (target: 14–18 PNGs):

| ID | Filename | Description | Source table |
|---|---|---|---|
| REGIME_50_01 | REGIME_50_01_<family>_n_by_regime.png | Bar chart: per-regime N per family | regime_metrics_long.csv |
| REGIME_50_02 | REGIME_50_02_<family>_mae_by_regime.png | Per-regime MAE per family per seed | regime_metrics_long.csv |
| REGIME_50_03 | REGIME_50_03_<family>_rmse_lift.png | RMSE lift heatmap (regime × seed) | regime_rmse_lift.csv |
| REGIME_50_04 | REGIME_50_04_change_magnitude_distribution.png | Train vs Test |Δy| distribution with Q90 line | (Train histogram pre-computed in Phase 50-B; Test histogram) |
| REGIME_50_05 | REGIME_50_05_target_level_train_threshold.png | Train raw Appliances distribution with Q25/Q75 lines | regime_thresholds_train_only.json + Train histogram |
| REGIME_50_06 | REGIME_50_06_extreme_high_threshold.png | Train raw Appliances distribution with Q90 line | regime_thresholds_train_only.json |
| REGIME_50_07 | REGIME_50_07_time_of_day_<model>.png | Time-of-day MAE/MAE_lift per model per regime | regime_metrics_long.csv |
| REGIME_50_08 | REGIME_50_08_day_type_<model>.png | Day-type MAE/MAE_lift per model per regime | regime_metrics_long.csv |
| REGIME_50_09 | REGIME_50_09_change_direction_<model>.png | Direction MAE/MAE_lift per model per regime | regime_metrics_long.csv |
| REGIME_50_10 | REGIME_50_10_rmse_lift_heatmap.png | Cross-seed RMSE lift heatmap by regime | regime_rmse_lift.csv aggregated |
| REGIME_50_11 | REGIME_50_11_persistence_vs_transformer.png | Per-regime MAE: Persistence vs Transformer | regime_persistence_delta_*.csv |
| REGIME_50_12 | REGIME_50_12_sae_contribution.png | SAE share by regime (stacked bar) | regime_metrics_long.csv |
| REGIME_50_13 | REGIME_50_13_sse_contribution.png | SSE share by regime (stacked bar) | regime_metrics_long.csv |
| REGIME_50_14 | REGIME_50_14_contrast_<contrast_id>.png | Pairwise contrast RMSE/MAE/MBE bars | regime_pairwise_contrasts.csv |
| REGIME_50_15 | REGIME_50_15_seed_spread_by_regime.png | seed_std_prediction by regime (boxplot) | regime_seed_spread_summary.csv |
| REGIME_50_16 | REGIME_50_16_seed_sign_consensus_by_regime.png | Stacked bar of consensus_class fractions by regime | regime_seed_sign_consensus_summary.csv |
| REGIME_50_17 | REGIME_50_17_train_vs_test_prevalence.png | Prevalence comparison bar | regime_train_vs_test_prevalence.csv |
| REGIME_50_18 | REGIME_50_18_lstm_status_placeholder.png | Single panel: "LSTM = NOT_EVALUATED_BY_PROTOCOL" | textual placeholder, no data plot |

**Report files**:
- `artifacts/error_by_regime/error_by_regime_findings.csv` (per-finding rows with safe descriptive wording)
- `artifacts/error_by_regime/error_by_regime_summary.json`
- `artifacts/error_by_regime/error_by_regime_report.md` (human-readable Markdown)
- `artifacts/error_by_regime/error_by_regime_README.md`
- `artifacts/error_by_regime/phase_50_signoff.json` (38+ gate checks)
- `artifacts/error_by_regime/phase50_handoff_to_phase51.json` (regime × target_id × per-seed error for Phase 51; no top-K)

---

## 19. Phase 51 handoff contract

`artifacts/error_by_regime/phase50_handoff_to_phase51.json`:

```json
{
  "phase": "50",
  "n_test": 2961,
  "seeds": [42, 123, 2026],
  "regime_assignment_sha256": "<from test_regime_assignment_fingerprint.json>",
  "residual_convention": "y_true - y_pred",
  "per_seed_per_target_regime_error": [
    {"target_id": "TGT_00016774", "regime_label_R1": "TL_LOW", "regime_label_R2": "NON_EXTREME",
     "regime_label_R3": "CHANGE_NORMAL", "regime_label_R4": "DIR_UP",
     "regime_label_R5": "TOD_MORNING", "regime_label_R6": "DAY_WEEKDAY",
     "seed": 42, "absolute_error_wh": 14.65, "squared_error_wh2": 214.82, "y_true_wh": 60.0, "y_pred_wh": 45.34},
    ...
  ],
  "forbidden_in_phase51": [
    "selecting a best seed",
    "creating an ensemble",
    "regime-specific recalibration",
    "re-deriving regime thresholds using Phase 51 worst cases"
  ],
  "phase51_first_phase_allowed_to_rank_worst_errors": true
}
```

---

## 20. Testing strategy (Phase 50-B..G)

10 test groups (per architecture amendment v1.12 §amendment_testing_strategy_lock_PROPOSED):

1. **TRAIN-ONLY**: 4 tests verifying Train-only threshold derivation.
2. **TEMPORAL**: 5 tests verifying gap-safe, no-split-crossing, exact 10-min, WB0 predecessor for Test first point.
3. **ASSIGNMENT**: 5 tests verifying test_regime_assignment.csv has no prediction fields, no residual fields, full coverage, explicit UNCLASSIFIED for R3/R4.
4. **JOIN**: 4 tests verifying many-to-one join, no row drop, no duplicate matches, join occurs after assignment freeze.
5. **METRICS**: 7 tests verifying MAE/RMSE/R² guard/MBE sign semantics/under-over-exact/SAE-SSE/rmse_lift_wh-vs-pct both present.
6. **CONTRIBUTION**: 4 tests verifying sample_share/SAE-share/SSE-share sum to 1 and global SSE reconstruction.
7. **CROSS-SEED**: 4 tests verifying all 3 seeds, ddof=1, no 3N iid, no best-seed.
8. **BASELINE**: 3 tests verifying Persistence same population, baseline-minus-transformer delta convention, LSTM remains NOT_ELIGIBLE.
9. **SCOPE**: 8 tests verifying no Test thresholds, no Cartesian mining, no correction, no training, no worst-error ranking, no attention analysis, no Phase 51 implementation, notebook unchanged except 2 appended cells.
10. **CONTRACT**: 3 tests verifying O50 inventory completeness, R² NOT_DEFINED serialization, regime sign consensus taxonomy 6-class.

Total target: ≥47 focused unit tests.

---

## 21. Discrepancy taxonomy

| Class | Example | Handling |
|---|---|---|
| MISSING_SOURCE | A canonical artifact is not at expected path | HALT Phase 50-B; require upstream fix |
| ARTIFACT_NAME_MISMATCH | Plan refers to `residual_bias_summary.csv`; actual is `phase49_signed_bias.csv` | Documented mapping; do NOT create alias |
| CONTRACT_AMBIGUITY | RMSELift ratio vs rmse_lift_wh | Pre-declare both fields with disambiguated names |
| DEGENERATE_THRESHOLD | Q25_y >= Q75_y due to discrete Appliances values | HALT R1; predeclared Train-only resolution per plan §13 |
| FINGERPRINT_MISMATCH | SHA256 of canonical artifact changed unexpectedly | HALT; investigate upstream |
| TRAIN_POPULATION_DRIFT | WINDOWPOP Train count != 13670 | HALT; investigate Phase 5 / Phase 10 / Phase 11 |
| LSTM_DRIFT | LSTM eligibility changes to ELIGIBLE | HALT; require separate human-approved Phase 50.5 for LSTM comparison |
| NOTEBOOK_DRIFT | CourseWork_1.ipynb cells other than Phase 49-G's 2 appended cells change | HALT Phase 50-H; investigate |

---

## 22. Forbidden actions (Phase 50-B onward — same as Phase 50-A)

- Compute Q25_y, Q75_y, Q90_y, Q90_abs_delta.
- Inspect Test error to design thresholds.
- Create train_regime_assignment.csv before Stage C freeze.
- Create test_regime_assignment.csv before Stage C freeze.
- Join residuals before test_regime_assignment.csv is frozen.
- Compute regime MAE/RMSE/R²/MBE.
- Compute regime SAE/SSE.
- Rank regimes by observed Test error.
- Compare regime model performance in any way that would motivate retuning.
- Generate Phase 50 scientific figures.
- Modify Phase 47 / 48 / 49 canonical artifacts.
- Modify Phase 47 / 48 / 49 signoffs.
- Run new Test inference.
- Load model checkpoints.
- Train.
- Call optimizer.step() / .backward() / model.train().
- Fit scalers.
- Select best seed.
- Create ensemble.
- Pool 3N residuals as iid.
- Create regime-specific correction.
- Create weighted Test headline metric.
- Perform worst-error ranking (Phase 51).
- Perform attention analysis (Phase 52+).
- Implement Phase 51.
- Modify CourseWork_1.ipynb (except appending 2 cells in Phase 50-H).

---

## 23. Definition of Done (Phase 50)

1. `architecture_rule.md` v1.12 has been Human-approved (or Phase 50 implementation has been explicitly authorized via Human instruction without v1.12 amendment).
2. `phase_50_error_by_regime_analysis_plan.md` (this file) has been Human-approved.
3. `phase_50_architecture_amendment_log.json` exists at `docs/save_log_in_processing/`.
4. All 39 O50 artifacts (`O50.1`..`O50.39`) are present and pass their respective audit checks.
5. `phase_50_signoff.json` is PASS with ≥38 gate checks passed.
6. `phase50_handoff_to_phase51.json` is emitted (Phase 51 can begin).
7. `CourseWork_1.ipynb` has exactly 2 new cells appended (markdown + dashboard code); all Phase 0–49 cells are byte-identical to the pre-Phase-50 backup.
8. `docs/save_log_in_processing/phase_50_error_by_regime_analysis_log.json` exists and is up-to-date.

---

## 24. Human approval gates

| Gate | Required before | Owner |
|---|---|---|
| Architecture amendment v1.12 | Phase 50-B implementation begins | Human |
| This pre-process plan v1.0 | Phase 50-B implementation begins | Human |
| Train-only thresholds (Q25/Q75/Q90/Q90_abs_delta) | Stage A freeze; before Stage B begins | Human (optional review; auto-approve if degenerate handling triggers) |
| Stage C test_regime_assignment.csv freeze | Stage D begins | Human (optional review; auto-approve if audit passes) |
| Phase 50 findings narrative | Phase 50-G signoff | Human |
| Phase 50 final signoff | Phase 51 begins | Human |
| Notebook Phase 50 cell append | Phase 50-H | Human |

---

## 25. Phase 50 sub-phase plan (B..H)

### 50-B — Train reference construction + threshold derivation
- **Inputs**: `data/raw_data/energydata_complete.csv`, `artifacts/windows/common_target_population.csv`, `artifacts/splits/split_manifest.json`, `artifacts/temporal/temporal_manifest.json`.
- **Modules**: `src/course_work/phase50/train_reference.py`, `src/course_work/phase50/thresholds.py`, `src/course_work/phase50/contract.py`.
- **Artifacts**: `regime_reference_train_manifest.json`, `regime_thresholds_train_only.json`, `regime_threshold_audit.csv`, `regime_threshold_fingerprint.json`, `phase50_preflight_audit.csv`, `phase50_b_manifest.json`, `error_by_regime_manifest.json`, `error_by_regime_contract.json`.
- **Tests**: 4 TRAIN-ONLY + 1 TEMPORAL + 3 CONTRACT.
- **Forbidden**: any read of Phase 49 residual artifacts; any Test access; any inference; any Phase 50-C onward artifacts.
- **Acceptance gate**: `regime_threshold_fingerprint.json.status == PASS` and Train target_ids_sha256 == expected hash from WINDOWPOP-v1 + split_id=TRAIN.
- **Human approval**: REQUIRED before 50-C.

### 50-C — Test regime assignment + freeze
- **Inputs**: `regime_thresholds_train_only.json` (FROZEN from 50-B), `artifacts/final_test/predictions/final_test_predictions_seed{42,123,2026}.csv` (read `y_true_wh` and `target_timestamp` and `target_id` only — NOT `y_pred_wh`), `artifacts/windows/common_target_population.csv` (for `continuity_segment_id` and `valid_L144`).
- **Modules**: `src/course_work/phase50/regime_assignment.py`, `src/course_work/phase50/assignment_audit.py`.
- **Artifacts**: `train_regime_assignment.csv`, `test_regime_assignment.csv`, `test_regime_assignment_audit.csv`, `test_regime_assignment_fingerprint.json`, `phase50_c_manifest.json`.
- **Tests**: 5 ASSIGNMENT + 1 TEMPORAL.
- **Forbidden**: any read of `residual_long_table.csv` or `residual_wide_table.csv`; any write to test_regime_assignment.csv after freeze; any new threshold computation; any prediction field in test_regime_assignment.csv.
- **Acceptance gate**: `test_regime_assignment_audit.csv` all 10 checks PASS + sha256 frozen.
- **Human approval**: REQUIRED before 50-D.

### 50-D — Residual join + per-seed regime metrics + contributions
- **Inputs**: `test_regime_assignment.csv` (FROZEN), `artifacts/residual_analysis/residual_long_table.csv`, `artifacts/residual_analysis/residual_wide_table.csv`.
- **Modules**: `src/course_work/phase50/error_join.py`, `src/course_work/phase50/metrics.py`, `src/course_work/phase50/contributions.py`.
- **Artifacts**: `regime_error_join_audit.csv`, `regime_metrics_long.csv`, `regime_metrics_target_level.csv`, `regime_metrics_extreme_high.csv`, `regime_metrics_change_magnitude.csv`, `regime_metrics_change_direction.csv`, `regime_metrics_time_of_day.csv`, `regime_metrics_day_type.csv`, `phase50_d_manifest.json`.
- **Tests**: 4 JOIN + 7 METRICS + 4 CONTRIBUTION.
- **Forbidden**: any threshold recomputation; any baseline metrics in this phase (50-F); any cross-seed aggregation in this phase (50-E); any sign-consensus or seed-spread regime aggregation (50-F).
- **Acceptance gate**: `regime_error_join_audit.csv` 0 unmatched + 0 duplicates + same_test_population=PASS + global SSE/SAE reconstruction matches per-seed Phase 47 metrics to ≤1e-9.
- **Human approval**: REQUIRED before 50-E.

### 50-E — Cross-seed aggregation + contrasts + rank stability + prevalence
- **Inputs**: `regime_metrics_long.csv` (from 50-D).
- **Modules**: `src/course_work/phase50/cross_seed.py`, `src/course_work/phase50/contrasts.py`, `src/course_work/phase50/rank_stability.py`, `src/course_work/phase50/prevalence.py`.
- **Artifacts**: `regime_cross_seed_summary.csv`, `regime_rmse_lift.csv`, `regime_pairwise_contrasts.csv`, `regime_rank_stability.csv`, `regime_train_vs_test_prevalence.csv`, `phase50_e_manifest.json`.
- **Tests**: 4 CROSS-SEED.
- **Forbidden**: any 3N iid pooling; any best-seed selection; any retuning; any new train/test error access.
- **Acceptance gate**: `regime_cross_seed_summary.csv` for each (family, label) has 3 seed-level rows and 1 cross-seed summary row (mean + sample SD with ddof=1).
- **Human approval**: REQUIRED before 50-F.

### 50-F — Persistence + LSTM placeholder + Phase 48 seed-spread by regime + Phase 49 sign-consensus by regime + baseline deltas
- **Inputs**: `regime_metrics_long.csv` (from 50-D), `artifacts/final_test/predictions/final_test_predictions_persistence.csv`, `artifacts/final_test/final_test_lstm_eligibility.json`, `artifacts/prediction_analysis/prediction_seed_spread.csv`, `artifacts/residual_analysis/phase49_cross_seed_sign_consensus.csv` (TAXONOMY ONLY — for the 6-class labels), `artifacts/residual_analysis/residual_long_table.csv` (for per-target per-seed sign computation).
- **Modules**: `src/course_work/phase50/persistence.py`, `src/course_work/phase50/lstm_context.py`, `src/course_work/phase50/seed_spread.py`, `src/course_work/phase50/sign_consensus.py`, `src/course_work/phase50/baseline_deltas.py`.
- **Artifacts**: `regime_metrics_persistence.csv`, `regime_metrics_lstm.csv`, `regime_persistence_delta_mae.csv`, `regime_persistence_delta_rmse.csv`, `regime_seed_spread_summary.csv`, `regime_seed_sign_consensus_summary.csv`, `baseline_delta_summary.csv`, `phase50_f_manifest.json`.
- **Tests**: 3 BASELINE.
- **Forbidden**: any LSTM residual fabrication; any new inference; any seed consensus threshold tuning; any best-seed selection.
- **Acceptance gate**: `regime_metrics_lstm.csv` has exactly 1 row with status=NOT_EVALUATED_BY_PROTOCOL; `regime_seed_sign_consensus_summary.csv` uses the canonical 6-class taxonomy.
- **Human approval**: REQUIRED before 50-G.

### 50-G — Figures + findings + discrepancies + report + README + Phase51 handoff + final signoff
- **Inputs**: All 50-B..F artifacts.
- **Modules**: `src/course_work/phase50/figures.py`, `src/course_work/phase50/findings.py`, `src/course_work/phase50/o50_inventory.py`, `src/course_work/phase50/handoff.py`, `src/course_work/phase50/signoff.py`, `src/course_work/phase50/finalization_documents.py`.
- **Artifacts**: `artifacts/error_by_regime/figures/*.png` (14–18 PNGs), `error_by_regime_findings.csv`, `error_by_regime_summary.json`, `error_by_regime_report.md`, `error_by_regime_README.md`, `error_by_regime_discrepancies.json`, `phase50_handoff_to_phase51.json`, `phase_50_signoff.json`, `phase50_g_manifest.json`.
- **Tests**: 10 SCOPE + 3 CONTRACT.
- **Forbidden**: any new analytical computation; any change to upstream artifacts; any worst-error ranking; any attention analysis; any notebook modification.
- **Acceptance gate**: `phase_50_signoff.json.status == PASS` with ≥38 gates passed; all 39 O50 artifacts present.
- **Human approval**: REQUIRED before 50-H.

### 50-H — Notebook presentation-only dashboard synchronization
- **Inputs**: `CourseWork_1.ipynb` (backup created first), `src/course_work/reporting/phase_50_dashboard.py` (NEW), all 50-G artifacts.
- **Modules**: `src/course_work/reporting/phase_50_dashboard.py`.
- **Notebook cells appended (exactly 2)**:
  - Markdown: `## Phase 50 - Error-by-Regime Analysis`
  - Code: `from course_work.reporting.phase_50_dashboard import render_phase_50_dashboard; display(render_phase_50_dashboard(PROJECT_ROOT))`
- **Forbidden**: any inline threshold computation; any `pd.read_csv` in notebook; any `def`/`class` in notebook; any materialize_phase50* call; any matplotlib inline figure; any modification to Phase 0–49 notebook cells.
- **Acceptance gate**: `nbclient`-executed Phase 50 cell succeeds; notebook backup SHA256 matches pre-modification cells; Phase 47/48/49 canonical artifacts unchanged.
- **Human approval**: REQUIRED before Phase 51 begins.

---

## 26. References

- Phase 50 plan (canonical detail): `docs/plan-doc/plan_detail_for_each_phase/Phase_50_Error-by-regime_analysis.md`
- Phase 49 plan (style reference): `docs/plan-doc/plan_before_process/phase_49_residual_analysis_plan.md`
- Phase 47 signoff: `artifacts/final_test/phase_47_signoff.json`
- Phase 48 signoff: `artifacts/prediction_analysis/phase_48_signoff.json`
- Phase 49 signoff: `artifacts/residual_analysis/phase_49_signoff.json`
- Phase 49 → Phase 50 handoff: `artifacts/residual_analysis/phase50_handoff.json`
- Architecture rule v1.11: `docs/RULE_BASE/architecture_rule.md` (current; Phase 50 NOT yet authorized)
- Architecture amendment v1.12 PROPOSED: `docs/save_log_in_processing/phase_50_architecture_amendment_log.json`
- Phase 50-A processing log: `docs/save_log_in_processing/phase_50_error_by_regime_analysis_log.json`
