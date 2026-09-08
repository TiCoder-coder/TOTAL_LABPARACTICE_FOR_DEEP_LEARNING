# Phase 47 — Corrective Post-Audit Test Re-evaluation Plan

**Document version:** `phase_47_corrective_post_audit_test_re_evaluation_plan.md`
**Created:** 2026-09-06
**Author:** Read-only audit + corrective governance (Part 2D / Part 2E)
**Classification:** Corrective pre-process plan — Phase 47
**Authority:** Companion to `phase_46_corrective_post_audit_reimplementation_plan.md`. Together they define the full corrective path Phase 46 → Phase 47 under the current Phase 45 lock.
**Status:** PLAN_READY_FOR_HUMAN_REVIEW
**Corrective Test access authorized:** NO

---

## 0. TL;DR

Phase 47 is a **corrective evaluation only**. It performs no training and no model selection. The model is already locked by Phase 45.

The prior Phase 47 Test access is classified:

```
PRIOR_PHASE_47_LINEAGE_STATUS: INVALIDATED_BY_PHASE46_LINEAGE_DRIFT
```

Prior Test evidence is preserved, not overwritten. Phase 47 may run only after:

1. Corrective Phase 46 has PASSED all 17 pre-Test gates (see `phase_46_corrective_post_audit_reimplementation_plan.md` §4).
2. Explicit human approval authorizes the corrective Test re-evaluation.

This plan defines the frozen Test contract, the required operations, the prior-evidence preservation procedure, the new evidence to be produced, and the corrective re-access event metadata.

---

## 1. What this plan governs

This plan governs only the corrective Phase 47 Test evaluation. It does **not** change:

- The Phase 45 lock.
- The Phase 46 corrective execution (governed by the sibling plan).
- Any upstream Phase 1-46 artifact.
- Any Test-driven decision (per `rule_code.md` Rule #76, Rule #77).

This plan **does** change:

- The Phase 47 evidence: a new `phase_47_signoff.json`, a new `final_test_summary.json`, a new `final_test_evaluation_contract.json`, a new `final_test_release_verification.json`, a new `final_test_access_event.json` (corrective re-access event), and new entries in `final_test_access_log.jsonl`.
- The prediction bundle CSV files: 4 new CSVs are persisted, replacing the current absent files.
- The downstream handoffs (Phase 48-52) that reference the new Phase 47 outputs.

The prior Phase 47 evidence (current `phase_47_signoff.json` created 2026-09-03T15:21:02, current `final_test_summary.json`, etc.) is preserved verbatim under `artifacts/final_test/_history/PHASE47_PRE_CORRECTIVE_<UTC>/`. **It is not modified, not overwritten, not deleted.**

---

## 2. Frozen Test contract

### 2.1 Test population (frozen, no recomputation)

```
Test N = 2961
Test population SHA = d7dbc0b3cde772dc3f62c181f0a09a4a7a5b0e7c78e567361dbfde089578da87
boundary_protocol = WB0_CONTEXT_CARRY_OVER
first_target_timestamp = 2016-05-07 04:40:00
last_target_timestamp  = 2016-05-27 18:00:00
target_ids_sha256 = d7dbc0b3cde772dc3f62c181f0a09a4a7a5b0e7c78e567361dbfde089578da87
horizon = 1
```

These values come from the current `final_test_population_manifest.json` (`status: PASS`) and are FROZEN. No window is added, removed, or relabeled.

### 2.2 Locked final model (from Phase 45, no change)

```
candidate_id        = TR_C2_ALT_LOOKBACK
lookback_steps      = 72
FINAL_REFIT_EPOCHS  = 30
seeds               = [42, 123, 2026]
config_fingerprint  = 585c5e79e6a1c8c49efdd2f7767e6d3d4c628835acfda360a2fa66a2ef5c1e24
final_lock_sha256   = 81fb87c44b6af31b6f65eff13956d9dc94a38dc75731a2c0af088228dd1bd4ec
```

`config_fingerprint` and `final_lock_sha256` are **two distinct identifiers** and MUST be stored and compared separately in every Phase 47 artifact. The Phase 47 evaluation contract must write `final_lock_sha256 = 81fb87c4...` (not the config_fingerprint) — this is the source-level fix described in §3 of the Phase 46 corrective plan.

### 2.3 Hard forbidden operations

The corrective Phase 47 run MUST NOT perform any of:

- Training or any optimizer step.
- Backward pass / gradient computation on Test.
- Scaler refit on Test.
- Best-seed selection based on Test metrics.
- Ensemble creation.
- Hyperparameter change.
- Architecture change.
- Lookback change.
- Target-scaling change.
- Feature-variant change.
- Seed-set change.
- Any Test-driven decision that feeds back into model selection.

---

## 3. Required corrective Phase 47 operations

### 3.1 Inputs

The corrective Phase 47 run must load:

- The corrected Phase 46 outputs (3 `.pt` checkpoint binaries + sidecars).
- The frozen Test population manifest (`final_test_population_manifest.json`).
- The frozen scalers (`final_scaling_contract.json` values, identical across all 3 seeds).
- The frozen evaluation contract (re-frozen before first Test y access).
- The `phase47_test_release.json` from the corrective Phase 46 PASS signoff (`released == true`).

### 3.2 Required operations

For each of seeds [42, 123, 2026]:

1. Load the corrected FINAL_REFIT checkpoint at the canonical path:
   - `artifacts/three_seed_final_runs/official_checkpoints/seed_42/seed_42_FINAL_REFIT.pt`
   - `artifacts/three_seed_final_runs/official_checkpoints/seed_123/seed_123_FINAL_REFIT.pt`
   - `artifacts/three_seed_final_runs/official_checkpoints/seed_2026/seed_2026_FINAL_REFIT.pt`
2. Verify strict-load compatibility (no shape mismatch, no config mismatch).
3. Run inference on the exact same Test IDs (2961 targets, no reordering).
4. Persist the prediction bundle CSV per seed:
   - `artifacts/final_test/predictions/final_test_predictions_seed42.csv`
   - `artifacts/final_test/predictions/final_test_predictions_seed123.csv`
   - `artifacts/final_test/predictions/final_test_predictions_seed2026.csv`
5. Compute per-seed MAE / RMSE / R² in raw Wh on the exact 2961 targets.
6. Compute mean and sample SD (ddof=1) across the 3 seeds.
7. Persist a Persistence baseline bundle:
   - `artifacts/final_test/predictions/final_test_predictions_persistence.csv`
8. Record per-CSV SHA256 in `prediction_checksums.json`.

In addition:

- LSTM_TUNED_DEV is **NOT_EVALUATED**. The frozen LSTM baseline uses lookback 144 while the final Transformer uses lookback 72, so the protocol is asymmetric. The eligibility file `final_test_lstm_eligibility.json` MUST preserve the status `NOT_ELIGIBLE_CONFIG_MISMATCH` with reason: `LSTM uses lookback=None, but final Transformer uses lookback=72` (the LSTM's effective lookback is 144 per `Phase_43_LSTM_tuning.md`; the Phase 47 baseline reason text is preserved as-is per existing convention).

---

## 4. Corrective re-access event metadata

The current active `final_test_access_event.json` has `event_id = "TEST_FIRST_ACCESS_EVENT"` (timestamp `2026-09-03T15:20:58`). The corrective run must record a **new** event documenting the documented re-access:

```json
{
  "event_id": "TEST_CORRECTIVE_RE_ACCESS_EVENT",
  "phase": 47,
  "authorized": true,
  "re_access_reason": "PHASE46_LINEAGE_INVALIDATED_BY_PROTOCOL_DEVIATION",
  "prior_access_event_id": "TEST_FIRST_ACCESS_EVENT",
  "prior_access_invalidated": true,
  "prior_phase47_status": "INVALIDATED_BY_PHASE46_LINEAGE_DRIFT",
  "final_lock_sha256": "81fb87c44b6af31b6f65eff13956d9dc94a38dc75731a2c0af088228dd1bd4ec",
  "config_fingerprint": "585c5e79e6a1c8c49efdd2f7767e6d3d4c628835acfda360a2fa66a2ef5c1e24",
  "evaluation_contract_sha256": "<computed at runtime>",
  "first_access_timestamp": "<computed at runtime>",
  "access_reason": "FINAL_HELD_OUT_EVALUATION_CORRECTIVE",
  "authorized_models": [
    "TRANSFORMER_SEED42",
    "TRANSFORMER_SEED123",
    "TRANSFORMER_SEED2026",
    "PERSISTENCE",
    "LSTM_TUNED_DEV_NOT_ELIGIBLE"
  ],
  "authorized_metrics": ["MAE_Wh", "RMSE_Wh", "R2"],
  "scientific_config_frozen": true,
  "post_access_tuning_forbidden": true,
  "no_model_change_since_lock": true,
  "no_decision_from_prior_test": true,
  "status": "AUTHORIZED_CORRECTIVE"
}
```

The new corrective event does NOT replace the prior event. The prior event is preserved in the archive.

---

## 5. New `phase_47_signoff.json` (corrective) — required fields

The new corrective signoff MUST include all of the following fields (in addition to the standard Phase 47 signoff fields):

| Field | Value |
|---|---|
| `phase` | `47` |
| `version` | `FINAL_TEST_EVAL-v2-CORRECTIVE` |
| `source_phase46_version` | `THREE_SEED_FINAL_RUNS-v2-CORRECTIVE` |
| `final_lock_sha256` | `81fb87c44b6af31b6f65eff13956d9dc94a38dc75731a2c0af088228dd1bd4ec` (canonical Phase 45 lock_sha, NOT config_fingerprint) |
| `config_fingerprint` | `585c5e79e6a1c8c49efdd2f7767e6d3d4c628835acfda360a2fa66a2ef5c1e24` (separate field, NOT substituted for lock_sha) |
| `evaluation_contract_sha256` | `<computed at runtime>` |
| `test_population_sha256` | `d7dbc0b3cde772dc3f62c181f0a09a4a7a5b0e7c78e567361dbfde089578da87` |
| `n_test` | `2961` |
| `seed_list` | `[42, 123, 2026]` |
| `seed42_checkpoint_sha256` | `<corrected Phase 46 seed 42 .pt SHA256>` |
| `seed123_checkpoint_sha256` | `<corrected Phase 46 seed 123 .pt SHA256>` |
| `seed2026_checkpoint_sha256` | `<corrected Phase 46 seed 2026 .pt SHA256>` |
| `seed42_mae_wh` | `<computed at runtime>` |
| `seed42_rmse_wh` | `<computed at runtime>` |
| `seed42_r2` | `<computed at runtime>` |
| `seed123_mae_wh` | `<computed at runtime>` |
| `seed123_rmse_wh` | `<computed at runtime>` |
| `seed123_r2` | `<computed at runtime>` |
| `seed2026_mae_wh` | `<computed at runtime>` |
| `seed2026_rmse_wh` | `<computed at runtime>` |
| `seed2026_r2` | `<computed at runtime>` |
| `transformer_mean_mae_wh` | `<computed at runtime>` |
| `transformer_sd_mae_wh` | `<computed at runtime>` |
| `transformer_mean_rmse_wh` | `<computed at runtime>` |
| `transformer_sd_rmse_wh` | `<computed at runtime>` |
| `transformer_mean_r2` | `<computed at runtime>` |
| `transformer_sd_r2` | `<computed at runtime>` |
| `persistence_mae_wh` | `<computed at runtime>` |
| `persistence_rmse_wh` | `<computed at runtime>` |
| `persistence_r2` | `<computed at runtime>` |
| `lstm_eligibility` | `NOT_ELIGIBLE_CONFIG_MISMATCH` |
| `lstm_metrics` | `null` |
| `all_common_targets_verified` | `true` |
| `best_seed_selected` | `false` |
| `ensemble_used` | `false` |
| `training_used` | `false` |
| `scaler_fit_used` | `false` |
| `post_test_tuning` | `false` |
| `prediction_bundles_frozen` | `true` |
| `prior_phase47_status` | `INVALIDATED_BY_PHASE46_LINEAGE_DRIFT` |
| `prior_phase47_artifacts_preserved` | `true` |
| `corrective_access_authorized_by` | `<human approver, recorded by name>` |
| `corrective_access_event_id` | `TEST_CORRECTIVE_RE_ACCESS_EVENT` |
| `ready_for_phase48` | `true` (only after PASS) |
| `ready_for_phase52` | `true` (only after PASS) |
| `warnings` | `[]` |
| `overall_status` | `PASS` (only after all gates pass) |

The new signoff is written to a separate filename to avoid collision with the prior archived signoff: `phase_47_corrective_signoff.json`. The standard `phase_47_signoff.json` is preserved (in `_history/`) but is NOT the authoritative file.

**DO NOT assume the prior Phase 47 numerical values will remain identical.** The corrected Phase 47 numbers are authoritative ONLY after they are generated from the corrected Phase 46 checkpoints (with the Phase 45 lock values) and verified by hash lineage at the prediction_bundle level.

---

## 6. Prior Phase 47 evidence preservation (BEFORE any corrective run starts)

Before the corrective Phase 47 run begins, the current active Phase 47 evidence must be archived. The archive destination and procedure follow the project's existing convention (see §5 of the Phase 46 corrective plan).

### 6.1 Phase 47 archive destination

```
COURSE_WORK/artifacts/final_test/_history/PHASE47_PRE_CORRECTIVE_<UTC>/
+ _archive_manifest.json
```

### 6.2 Files to preserve (Phase 47)

| Source path | Status |
|---|---|
| `phase_47_signoff.json` | CURRENT (numerically consistent but lineage-invalid); preserved. |
| `final_test_summary.json` | CURRENT; preserved. |
| `final_test_evaluation_contract.json` | CURRENT; preserved (note: contains the `final_lock_sha256 = 585c5e79...` conflation; preserved as evidence of the bug). |
| `final_test_evaluation_manifest.json` | CURRENT; preserved. |
| `final_test_release_verification.json` | CURRENT (references missing `phase47_test_release.json`); preserved. |
| `final_test_population_manifest.json` | CURRENT; preserved (Test population manifest is FROZEN, applies to both prior and corrective runs). |
| `final_test_access_event.json` | CURRENT (`event_id = TEST_FIRST_ACCESS_EVENT`); preserved. |
| `final_test_access_log.jsonl` | CURRENT (4 inference events); preserved (append-only). |
| `final_test_lstm_eligibility.json` | CURRENT (`NOT_ELIGIBLE_CONFIG_MISMATCH`); preserved. |
| `final_test_discrepancies.json` | CURRENT (`discrepancy_count: 0` — not valid post-corrective); preserved. |
| `prediction_checksums.json` | CURRENT (referenced but CSV files absent); preserved. |
| `phase48_prediction_analysis_handoff.json` | CURRENT; preserved. |
| `phase49_residual_analysis_handoff.json` | CURRENT; preserved. |
| `phase50_error_regime_handoff.json` | CURRENT; preserved. |
| `phase51_worst_error_handoff.json` | CURRENT; preserved. |
| `phase52_attention_extraction_handoff.json` | CURRENT; preserved. |
| `final_test_report.md` | CURRENT; preserved. |
| `README_FINAL_TEST_EVALUATION.md` | CURRENT; preserved. |
| `.archive/*` | 87 historical test access events; UNTOUCHED. |
| `figures/*` | CURRENT; preserved. |

### 6.3 Archive procedure (safe by construction)

For each file in §6.2:

1. Compute SHA256 of source.
2. Copy (not move) into `artifacts/final_test/_history/PHASE47_PRE_CORRECTIVE_<UTC>/`.
3. Compute SHA256 of the archived copy.
4. Require `source_sha == archived_sha`.
5. Append to `_archive_manifest.json` with `reason = "PHASE47_PRE_CORRECTIVE"`.

**Rollback on any SHA mismatch.** The `.archive/` directory of historical test access events is NEVER modified.

### 6.4 NEW: Archive script for Phase 47 evidence

A new script `src/course_work/scripts/p47_archive_current_evidence.py` will be created (after code-implementation review per §3 of the Phase 46 corrective plan), mirroring the existing `p46_archive_historical_checkpoints.py` for Phase 47 evidence. **No archive script has been edited in this Part 2E task.**

---

## 7. New corrective Phase 47 outputs

### 7.1 Persisted prediction bundle CSVs (NEW — currently absent)

```
COURSE_WORK/artifacts/final_test/predictions/final_test_predictions_seed42.csv
COURSE_WORK/artifacts/final_test/predictions/final_test_predictions_seed123.csv
COURSE_WORK/artifacts/final_test/predictions/final_test_predictions_seed2026.csv
COURSE_WORK/artifacts/final_test/predictions/final_test_predictions_persistence.csv
```

### 7.2 New / rewritten collective outputs

```
COURSE_WORK/artifacts/final_test/prediction_checksums.json                       (rewritten with new per-seed SHAs)
COURSE_WORK/artifacts/final_test/final_test_summary.json                         (rewritten)
COURSE_WORK/artifacts/final_test/final_test_evaluation_contract.json             (rewritten with final_lock_sha256=81fb87c4..., NOT config_fingerprint)
COURSE_WORK/artifacts/final_test/final_test_evaluation_manifest.json             (rewritten)
COURSE_WORK/artifacts/final_test/final_test_release_verification.json            (rewritten against corrected phase47_test_release.json)
COURSE_WORK/artifacts/final_test/final_test_access_event.json                    (rewritten as TEST_CORRECTIVE_RE_ACCESS_EVENT)
COURSE_WORK/artifacts/final_test/final_test_access_log.jsonl                     (append corrective entries)
COURSE_WORK/artifacts/final_test/final_test_discrepancies.json                   (rewritten)
COURSE_WORK/artifacts/final_test/final_test_lstm_eligibility.json                (preserved verbatim: NOT_ELIGIBLE_CONFIG_MISMATCH)
COURSE_WORK/artifacts/final_test/phase_47_corrective_signoff.json                (NEW authoritative signoff)
COURSE_WORK/artifacts/final_test/phase48_prediction_analysis_handoff.json        (rewritten with new run_ids and SHAs)
COURSE_WORK/artifacts/final_test/phase49_residual_analysis_handoff.json          (rewritten)
COURSE_WORK/artifacts/final_test/phase50_error_regime_handoff.json               (rewritten)
COURSE_WORK/artifacts/final_test/phase51_worst_error_handoff.json                (rewritten)
COURSE_WORK/artifacts/final_test/phase52_attention_extraction_handoff.json       (rewritten)
COURSE_WORK/artifacts/final_test/final_test_report.md                            (rewritten)
```

### 7.3 Final test population manifest — FROZEN, not rewritten

`artifacts/final_test/final_test_population_manifest.json` is **FROZEN** and applies to both the prior and corrective runs. It is preserved unchanged in the new active location AND archived. The corrective Phase 47 must use the same 2961 targets.

---

## 8. Per-seed metrics output schema

Each per-seed entry in `final_test_summary.json` MUST contain:

```json
{
  "model_id": "TRANSFORMER_SEED<seed>",
  "seed": <seed>,
  "run_id": "<corrected Phase 46 run_id>",
  "checkpoint_sha256": "<corrected Phase 46 .pt SHA256>",
  "checkpoint_path": "artifacts/three_seed_final_runs/official_checkpoints/seed_<seed>/seed_<seed>_FINAL_REFIT.pt",
  "n_samples": 2961,
  "mae_wh": <float>,
  "rmse_wh": <float>,
  "r2": <float>,
  "r2_status": "DEFINED",
  "population_sha256": "d7dbc0b3cde772dc3f62c181f0a09a4a7a5b0e7c78e567361dbfde089578da87"
}
```

Seed aggregation:

```json
{
  "seed_aggregation_semantics": "MEAN_PLUS_SAMPLE_SD",
  "mae_wh": { "seed42": ..., "seed123": ..., "seed2026": ..., "mean": ..., "sample_sd": ..., "min": ..., "max": ..., "range": ... },
  "rmse_wh": { ... },
  "r2": { ... }
}
```

Sample SD is computed with `ddof=1`.

---

## 9. Stop conditions

STOP and DO NOT proceed if any of the following occur:

1. The corrective Phase 46 has not PASSED all 17 pre-Test gates (see `phase_46_corrective_post_audit_reimplementation_plan.md` §4).
2. `phase47_test_release.json` does not exist or `released != true`.
3. The corrective Phase 46 signoff is not PASS.
4. Any corrected Phase 46 `.pt` checkpoint binary is missing or its SHA256 does not match metadata.
5. The Test population SHA does not equal `d7dbc0b3cde772dc3f62c181f0a09a4a7a5b0e7c78e567361dbfde089578da87`.
6. Test N is not 2961.
7. Any inference output is computed against a target ID that is not in the frozen Test population.
8. Any scaler refit is attempted on Test.
9. Any training or optimizer step is attempted.
10. Any best-seed selection, ensemble creation, or post-Test tuning is attempted.
11. The prior Phase 47 archive is not complete (any missing SHA verification).

---

## 10. Human approval gates

Per `rule_code.md` Rule #10-#14 and Rule #76, no Test access proceeds without explicit human approval:

1. **Human approval #1:** Review of this plan + the root-cause audit + the Phase 46 corrective plan. Authorizes code implementation per §3 of the Phase 46 corrective plan.
2. **Human approval #2:** Review of the implementation diff. Authorizes the corrective Phase 46 training run.
3. **Human approval #3:** Review of the corrective Phase 46 PASS signoff (with all 17 gates passing). Authorizes corrective Phase 47 Test re-evaluation under this plan.

Each approval is explicit and recorded in `phase_47_corrective_signoff.json.corrective_access_authorized_by`.

---

## 11. Why the prior Phase 47 lineage is invalid

(Detailed in `phase_46_47_lineage_drift_audit_2026_09_06.md` §3-§7; summarized here for self-contained governance.)

1. Phase 46 active artifacts reference `TR_C0_PRIMARY` / lookback 36 / 50 epochs / `config_sha256 = a711a9b8...` / `final_lock_sha256 = a711a9b8...` (== config_sha). None of these match the current Phase 45 lock (`TR_C2_ALT_LOOKBACK` / 72 / 30 / `config_fingerprint = 585c5e79...` / `final_lock_sha256 = 81fb87c4...`).
2. Phase 46 actual `.pt` checkpoint binaries are absent from the repository. Only metadata sidecars exist.
3. Phase 47 references per-seed checkpoint SHA256s and run_ids (`RUN_TR_FSD_0254_*` / `RUN_TR_FSD_0255_*` with `c3cfad11...` / `8a134fec...` / `875380053...`) that do NOT appear in the experiment registry, as binary files, or in any training log.
4. Phase 47 `final_lock_sha256` field is set to the Phase 45 **config_fingerprint** (`585c5e79...`) — not the canonical Phase 45 **final_lock_sha256** (`81fb87c4...`). This conflation is documented by `phase58_active_revision_audit_checkpoint.json` (`FINAL_LOCK_SHA_CORRECT: false`).
5. Phase 47 prediction bundle CSVs and `phase47_test_release.json` are absent from the repository.

The prior numerical Phase 47 metrics are internally consistent and match the expected values, but the lineage `final_lock_sha256 → config_fingerprint → checkpoint_sha256 → prediction_bundle_sha256 → metric` cannot be reconstructed from the active evidence. **Numerical consistency is not lineage validity.**

---

## 12. Status

```
STATUS: PLAN_READY_FOR_HUMAN_REVIEW
CORRECTIVE_TEST_ACCESS_AUTHORIZED: NO

PHASE_47_TYPE: CORRECTIVE_EVALUATION_ONLY
PHASE_47_TRAINING: FORBIDDEN
PHASE_47_MODEL_SELECTION: FORBIDDEN
PRIOR_PHASE_47_LINEAGE_STATUS: INVALIDATED_BY_PHASE46_LINEAGE_DRIFT
PRIOR_PHASE_47_NUMERICS_STATUS: PRESERVED_AS_HISTORICAL_EVIDENCE
PRIOR_PHASE_47_ARCHIVE_DESTINATION: artifacts/final_test/_history/PHASE47_PRE_CORRECTIVE_<UTC>/
ARCHIVE_EXECUTED: NO
TEST_ACCESS_OCCURRED: NO (in this Part 2E task)

CONFIG_FINGERPRINT: 585c5e79e6a1c8c49efdd2f7767e6d3d4c628835acfda360a2fa66a2ef5c1e24
FINAL_LOCK_SHA256:  81fb87c44b6af31b6f65eff13956d9dc94a38dc75731a2c0af088228dd1bd4ec
TEST_POPULATION_SHA: d7dbc0b3cde772dc3f62c181f0a09a4a7a5b0e7c78e567361dbfde089578da87
TEST_N: 2961
```

---

**This plan is documentation only. It does not authorize any Test access.**
