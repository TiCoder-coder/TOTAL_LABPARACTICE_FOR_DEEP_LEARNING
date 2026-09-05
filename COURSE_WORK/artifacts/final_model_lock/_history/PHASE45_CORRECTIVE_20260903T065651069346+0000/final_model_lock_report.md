# Phase 45 — Final Model Lock Report

**Phase**: 45 (Final Model Lock / FINAL_MODEL_LOCK-v1)
**Generated**: 2026-09-03T06:46:08.368040+00:00

## 1. Locked Candidate
- candidate_id: `TR_C2_ALT_LOOKBACK`
- model_family: `TRANSFORMER_ENCODER`
- lookback_steps: `72`
- feature_variant: `FS2_TF1`
- target_scaling: `YS1`
- config_fingerprint: `585c5e79e6a1c8c49efdd2f7767e6d3d4c628835acfda360a2fa66a2ef5c1e24`

## 2. Final Epochs
- RO1 inner best epoch: `30`
- RO2 inner best epoch: `37`
- RO3 inner best epoch: `19`
- sorted: `[19, 30, 37]`
- FINAL_REFIT_EPOCHS (median): **`30`**
- aggregation rule: `MEDIAN_RO_INNER_BEST_EPOCHS-v1`
- candidate max_epochs: `50`
- within cap: `True`

## 3. FINAL_DEV_REGION-v1
- target_count: `16774`
- first_target_timestamp: `2016-01-11 17:00:00`
- last_target_timestamp: `2016-05-07 04:30:00`
- first_test_timestamp: `2016-05-07 04:40:00`
- target_ids_fingerprint: `552e6895dbc6f7a93b0ea9a2ab4de77abd58d8780ad4bc85b697a485ccbc31a9`

## 4. Phase44 Recommendation
- pooled_rmse_wh: `59.85979463604273`
- pooled_mae_wh: `27.950651178167792`
- pooled_r2: `0.5788465716025317`

## 5. Fingerprints
- FINAL_MODEL_CONFIG_SHA256: `585c5e79e6a1c8c49efdd2f7767e6d3d4c628835acfda360a2fa66a2ef5c1e24`
- FINAL_TRAINING_RECIPE_SHA256: `857dbaf7903792cdba3e126a50d8919992f02e0fff838cba86180026c9e0220c`
- FINAL_LINEAGE_SHA256: `9ba532539d548156e1d5d933abe8c1463617cdc790741e48d5249b452995d9fe`
- FINAL_MODEL_LOCK_SHA256: `81fb87c44b6af31b6f65eff13956d9dc94a38dc75731a2c0af088228dd1bd4ec`

## 6. Acceptance Checks
- phase44_signoff_overall_status: PASS  (severity=CRITICAL)
- approved_for_phase45_true: PASS  (severity=CRITICAL)
- test_status_not_accessed: PASS  (severity=CRITICAL)
- final_family_transformer_only: PASS  (severity=CRITICAL)
- candidate_fingerprint_match: PASS  (severity=CRITICAL)
- three_inner_epochs_present: PASS  (severity=CRITICAL)
- median_rule_exact: PASS  (severity=CRITICAL)
- no_fallback_50: PASS  (severity=CRITICAL)
- lookback_72: PASS  (severity=CRITICAL)
- final_dev_excludes_test: PASS  (severity=CRITICAL)
- last_final_dev_ts_before_first_test_ts: PASS  (severity=CRITICAL)
- final_scaler_excludes_test: PASS  (severity=CRITICAL)
- final_seeds_exact: PASS  (severity=CRITICAL)
- three_planned_phase46_runs: PASS  (severity=CRITICAL)
- same_epochs_all_seeds: PASS  (severity=CRITICAL)
- phase46_validation_none: PASS  (severity=CRITICAL)
- early_stopping_false: PASS  (severity=CRITICAL)
- checkpoint_type_final_refit: PASS  (severity=CRITICAL)
- no_best_semantics: PASS  (severity=CRITICAL)
- attention_compatibility_retained: PASS  (severity=CRITICAL)
- wb0_locked_no_amendment: PASS  (severity=CRITICAL)
- zero_training_zero_validation: PASS  (severity=CRITICAL)
- o45_artifacts_38_of_38: PASS  (severity=CRITICAL)
- phase46_handoff_complete: PASS  (severity=CRITICAL)
- phase47_guard_complete: PASS  (severity=CRITICAL)

## 7. Phase46 + Phase47 Handoff
- Planned Phase46 runs: `FINAL_TS_SEED_42`, `FINAL_TS_SEED_123`, `FINAL_TS_SEED_2026`
- Phase47 test-evaluation guard: `phase47_test_evaluation_guard.json`
- validation_loader: NONE
- early_stopping: false
- checkpoint_type: FINAL_REFIT

## 8. Safety
- optimizer steps in this run: `0`
- new scientific RUN IDs: `0`
- new validation runs: `0`
- Test access: NO
