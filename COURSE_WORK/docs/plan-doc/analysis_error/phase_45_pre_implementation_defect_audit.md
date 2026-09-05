# Phase 45 Pre-Implementation Defect Audit

**Source**: One-shot Phase 45 audit (39 defects identified)
**Created**: 2026-09-03 (Phase 45 corrective implementation baseline)
**Status**: PRE-FIX snapshot. See `phase_45_post_implementation_closure.md` for actual closure state.

---

## Defect Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 28 |
| MAJOR    | 8 |
| MINOR    | 3 |
| **TOTAL** | **39** |

---

## Closure Outcome (Post-Implementation Re-Audit)

| Severity | Fixed | Not_Fixed | Not_Applicable |
|----------|-------|-----------|----------------|
| CRITICAL | 28    | 0         | 0              |
| MAJOR    | 8     | 0         | 0              |
| MINOR    | 3     | 0         | 0              |
| **TOTAL** | **39** | **0** | **0** |

---

## Defect Matrix

### CRITICAL (28)

| ID | Severity | Plan Section | Root Cause | File/Function | Scientific Impact | Required Fix | Test Required | Status     |
|----|----------|--------------|------------|---------------|-------------------|--------------|---------------|--------|
| C1 | CRITICAL | §2 §4 | Script reads non-existent handoff keys `locked_model_id`, `config`, `config_fingerprint` | `scripts/phase45_final_model_lock.py::main` | Phase45 crashes immediately with `KeyError` | Use correct keys `recommended_transformer_*` | Yes | FIXED    |
| C2 | CRITICAL | §4 | Checks `phase_44_signoff.status` but actual field is `overall_status` | `phase45_final_model_lock.py::main` | Preflight gate silently fails or passes incorrectly | Use `overall_status`; also verify `approved_for_phase45` | Yes | FIXED    |
| C3 | CRITICAL | §56–§60 | `final_refit_epochs = max_epochs` (fallback=50) — explicitly forbidden | `phase45_final_model_lock.py` | Falls back to forbidden epoch policy | Compute `median(RO1,RO2,RO3)` from handoff evidence | Yes | FIXED    |
| C4 | CRITICAL | §102 | `rolling_origin_inner_best_epochs.csv` has NaN in 6 columns | `src/course_work/rolling_origin/real_run.py:1170-1174` | Phase45 cannot verify `best_inner_rmse_wh` / `sample_count_inner_val` / fingerprint | Populate full rows from `StageAResult` (no retrain) | Yes | FIXED    |
| C5 | CRITICAL | §102 | `write_inner_best_epochs` header expects 9 fields but writer passes only 3 | `src/course_work/rolling_origin/artifacts.py:316-330` | CSV produced with empty cells | Either populate rows in real_run.py or shrink writer header | Yes | FIXED    |
| C6 | CRITICAL | §98 | No `--mode` argparse / no CLI mode | `phase45_final_model_lock.py::main` | Cannot run audit-only / lock-only / prelock-gate modes | Add argparse with `--mode lock`/`audit`/`prelock`/`dry-run` | Yes | FIXED    |
| C7 | CRITICAL | §183–§192 | No plan-conformance gate at all | `phase45_final_model_lock.py::main` | Cannot reject on lineage / WB0 / FINAL_DEV violations | Implement the full plan conformance gate | Yes | FIXED    |
| C8 | CRITICAL | §6 §10 | Hard-codes `"TRANSFORMER_ENCODER"` without verifying recommended candidate model_family | `phase45_final_model_lock.py` | Could lock LSTM if family check skipped | Read `recommended_transformer_config.model.model_family` and ASSERT == `TRANSFORMER_ENCODER` | Yes | FIXED    |
| C9 | CRITICAL | §12 §25 | Hard-codes `protocol_amendment_required: False` without S19 evidence | `phase45_final_model_lock.py` | Cannot detect pending amendments | Load S19 artifact; assert `protocol_amendment_required == False` (FAIL otherwise) | Yes | FIXED    |
| C10 | CRITICAL | §103 | No `final_lineage_audit.csv` written | `phase45_final_model_lock.py` | Lineage cannot be verified | Build lineage audit (S1–S19 + Phase42 + Phase44) | Yes | FIXED    |
| C11 | CRITICAL | §102 | No `final_candidate_source_audit.csv` written | `phase45_final_model_lock.py` | Source field provenance cannot be verified | Build candidate-source audit (Phase42 vs Phase44 vs Phase45) | Yes | FIXED    |
| C12 | CRITICAL | §98 O45.3 | `final_model_lock_audit.csv` is 4 rows instead of full preflight audit | `phase45_final_model_lock.py:67-69` | Insufficient preflight evidence | Emit full preflight audit (Phase44 status, lineage, candidate, epochs, FINAL_DEV, scaler) | Yes | FIXED    |
| C13 | CRITICAL | §25–§27 | `FINAL_DEV_REGION-v1` written with None / static fields; no actual target_ids built | `phase45_final_model_lock.py:230-251` | Phase46 will not have a deterministic population | Build `FINAL_DEV_TARGET_IDS` from canonical windowpop; compute fingerprint | Yes | FIXED    |
| C14 | CRITICAL | §29 | `final_scaling_contract` lacks explicit `REQUIRED_AT_PHASE46` markers for unfitted scalers | `phase45_final_model_lock.py:252-263` | Phase46 may invent checksums | Either fit Phase9 scalers on FINAL_DEV (no-train) or mark checksums `REQUIRED_AT_PHASE46` | Yes | FIXED    |
| C15 | CRITICAL | §83 §172 | No `--seed` flag | `phase45_final_model_lock.py::main` | Seed list handling undefined | Add `--seed` (default 42) for deterministic ordering | Yes | FIXED    |
| C16 | CRITICAL | §84 | `test_status` field present in signoff but no assertion on canonical Phase44 `test_status == NOT_ACCESSED` | `phase45_final_model_lock.py:395-399` | Test leakage not gated | Read Phase44 `test_status`; FAIL if not NOT_ACCESSED | Yes | FIXED    |
| C17 | CRITICAL | §105 | `final_feature_contract.feature_names` is empty `[]` | `phase45_final_model_lock.py:156` | Cannot verify feature ordering | Load feature names from candidate (or LSTM context which shares FS2_TF1) | Yes | FIXED    |
| C18 | CRITICAL | §118 O45.20 | No `final_attention_compatibility_audit.csv` | `phase45_final_model_lock.py` | Attention support cannot be verified | Implement attention audit per-row checks | Yes | FIXED    |
| C19 | CRITICAL | §127 O45.29 | No `baseline_context_evidence.json` | `phase45_final_model_lock.py` | LSTM/Persistence context lost | Build from Phase44 handoff `lstm_tuned_context` + `persistence_context` | Yes | FIXED    |
| C20 | CRITICAL | §126 O45.28 | No `boundary_sensitivity_evidence.json` | `phase45_final_model_lock.py` | WB0 sensitivity cannot be evidenced | Build from S19 sweep artifact; verify WB0 primary | Yes | FIXED    |
| C21 | CRITICAL | §129 O45.31 | No `phase47_test_evaluation_guard.json` | `phase45_final_model_lock.py` | Test-guard missing | Write guard with `test_access_first_allowed_phase=47` | Yes | FIXED    |
| C22 | CRITICAL | §130 O45.32 | No `final_model_lock_findings.csv` | `phase45_final_model_lock.py` | Finding codes missing | Emit per-code findings | Yes | FIXED    |
| C23 | CRITICAL | §131 O45.34 | No `final_model_lock_discrepancies.json` | `phase45_final_model_lock.py` | Discrepancies missing | Emit array | Yes | FIXED    |
| C24 | CRITICAL | §133 O45.33 | No `final_model_lock_tests.csv` | `phase45_final_model_lock.py` | Acceptance-check evidence missing | Emit per-check rows | Yes | FIXED    |
| C25 | CRITICAL | §98 O45.3 | No `phase45_preflight_audit.csv` | `phase45_final_model_lock.py` | Preflight evidence insufficient | Write full preflight audit | Yes | FIXED    |
| C26 | CRITICAL | §125 O45.27 | No `rolling_origin_selection_evidence.json` | `phase45_final_model_lock.py` | Phase44 selection rationale missing | Build from Phase44 fold/pooled/ranking files | Yes | FIXED    |
| C27 | CRITICAL | §150 O45.36 | No `final_model_lock_report.md` | `phase45_final_model_lock.py` | Human-readable report missing | Emit 22-section report | Yes | FIXED    |
| C28 | CRITICAL | §98 | Idempotency broken: timestamps inside fingerprint inputs | `phase45_final_model_lock.py` | Same inputs → different hashes (re-runs fail determinism check) | Strip timestamps from all fingerprint input payloads | Yes | FIXED    |

### MAJOR (8)

| ID | Severity | Plan Section | Root Cause | File/Function | Scientific Impact | Required Fix | Test Required | Status     |
|----|----------|--------------|------------|---------------|-------------------|--------------|---------------|--------|
| M1 | MAJOR | §177 | No archival of stale Phase45 artifacts before overwrite | `phase45_final_model_lock.py` | Cannot recover pre-correction state | Archive to `_history/PHASE45_CORRECTIVE_<UTC>/` with manifest | Yes | FIXED    |
| M2 | MAJOR | §98 | `final_training_recipe_fingerprint.json` source path is wrong relative path | `phase45_final_model_lock.py:332` | Source tracing broken | Use canonical relative path | Yes | FIXED    |
| M3 | MAJOR | §83 | `final_model_lock_summary.json` uses `phase_id: 45` as int — schema drift from Phase44 | `phase45_final_model_lock.py:357` | Cross-phase loaders may break | Match Phase44 schema (string or int — both accepted) | Yes | FIXED    |
| M4 | MAJOR | §84 | `final_checkpoint_contract.official_epoch` uses `max_epochs` not derived `FINAL_REFIT_EPOCHS` | `phase45_final_model_lock.py:283` | Phase46 will train wrong number of epochs | Use derived median epoch | Yes | FIXED    |
| M5 | MAJOR | §127 | `phase46_three_seed_handoff.target_ids_fingerprint` is None | `phase45_final_model_lock.py:412` | Phase46 cannot validate population | Populate with FINAL_DEV fingerprint | Yes | FIXED    |
| M6 | MAJOR | §156 | `phase46_three_seed_handoff` lacks `FINAL_REFIT_MODE-v1` semantic field | `phase45_final_model_lock.py:409-430` | Phase46 contract incomplete | Add `final_refit_mode` field | Yes | FIXED    |
| M7 | MAJOR | §98 | Hardcoded handoff paths (no DI) | `phase45_final_model_lock.py` | Test isolation impossible | Make paths configurable via flags | Yes | FIXED    |
| M8 | MAJOR | §4 | No canonical Phase44 source-of-truth reader; inconsistent field access | `phase45_final_model_lock.py` | Drift between consumers | Build single handoff loader that asserts schema | Yes | FIXED    |

### MINOR (3)

| ID | Severity | Plan Section | Root Cause | File/Function | Scientific Impact | Required Fix | Test Required | Status     |
|----|----------|--------------|------------|---------------|-------------------|--------------|---------------|--------|
| m1 | MINOR | §151 | README is 1-line placeholder | `phase45_final_model_lock.py:363` | Insufficient documentation | Multi-section README | No | FIXED    |
| m2 | MINOR | §98 | `now_iso()` has dead branch `if "datetime" in globals()` | `phase45_final_model_lock.py:38` | Cosmetic | Clean up | No | FIXED    |
| m3 | MINOR | §172 | `DEVELOPMENT_SEED` import unused in main flow | `phase45_final_model_lock.py` | No reproducibility anchor | Use it for deterministic audit ordering | No | FIXED    |

---

## Plan-Conformance Snapshot

| Plan § | Conformance |
|--------|-------------|
| §2 no-training | PASS (script never trains) but crashes immediately |
| §4 Phase44 inputs | FAIL — wrong field names |
| §6 Transformer-only | FAIL — hard-coded without verification |
| §7-9 candidate lock | FAIL — placeholder |
| §10-17 final config fields | PARTIAL |
| §25-27 FINAL_DEV | FAIL — static metadata with None |
| §29 final scaler | FAIL — no checksums / no REQUIRED_AT_PHASE46 |
| §30-34 epoch policy | FAIL — median not computed |
| §39-41 seeds | PASS — [42,123,2026] written |
| §56-60 fingerprints | PARTIAL — non-deterministic |
| §83-93 run-matrix / handoff / guard | FAIL |
| §98-99 O45.* artifacts | FAIL — only ~14/38 produced |
| §183-192 acceptance checklist | FAIL |

---

## Safety Snapshot (Pre-Correction)

```
Optimizer steps:        0
New scientific RUN IDs: 0
New validation runs:    0
Test access:            NO
Phase46 executed:       NO
Phase44 evidence:       UNTOUCHED
```
