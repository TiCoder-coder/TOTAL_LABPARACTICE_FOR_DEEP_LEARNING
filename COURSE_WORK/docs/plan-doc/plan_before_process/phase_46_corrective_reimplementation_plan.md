# PHASE 46 — CORRECTIVE REIMPLEMENTATION PLAN

**Document version:** `phase_46_corrective_reimplementation_plan.md`
**Created:** 2026-09-01
**Author:** Senior ML Engineer / Scientific Reproducibility Engineer
**Classification:** Corrective Implementation Plan — Phase 46
**Authority:** Supersedes previous Phase 46 implementation

---

## 1. Implementation Boundaries

### 1.1 MAY Change

- Create new source files in `src/course_work/data/` (FINAL_DEV, FINAL_SCALING)
- Create new source files in `src/course_work/scaling/` (FINAL_SCALING-v1)
- Modify `scripts/phase46_three_seed_runs.py` (driver corrections)
- Modify `src/course_work/training/engine.py` (FINAL_REFIT semantics)
- Create new test files in `tests/unit/` and `tests/integration/`
- Create new artifact-generation logic for Phase 46 outputs
- Create new preflight/dry-run mode
- Create new documentation

### 1.2 MUST NOT Change

- Historical Phase 46 run IDs/checkpoints (RUN_TR_FSD_0153, 0154, 0155)
- Historical Phase 46 artifacts (status.json, training_history.csv, checkpoints)
- Phase 9 scaler files (`artifacts/scalers/x/`, `artifacts/scalers/y/`)
- Phase 9 scaling infrastructure (`src/course_work/data/scaling.py`)
- Phase 45 final model lock artifacts
- Any Test artifacts
- Git history

---

## 2. Source Files Expected to Change

### New Files

| File | Purpose |
|------|---------|
| `src/course_work/scaling/final_scaling.py` | FINAL_SCALING-v1 implementation |
| `src/course_work/data/final_dev.py` | FINAL_DEV_REGION-v1 construction |
| `src/course_work/training/final_refit.py` | FINAL_REFIT engine semantics |
| `tests/unit/test_final_dev.py` | FINAL_DEV unit tests |
| `tests/unit/test_final_scaling.py` | FINAL_SCALING unit tests |
| `tests/integration/test_phase46_preflight.py` | Phase 46 preflight integration tests |
| `scripts/phase46_preflight.py` | Phase 46 dry-run/preflight script |

### Modified Files

| File | Changes |
|------|---------|
| `scripts/phase46_three_seed_runs.py` | Wire FINAL_DEV + FINAL_SCALING, add assertions, fix phase47 release |
| `src/course_work/training/engine.py` | Fix best_epoch for evaluate_validation=False |
| `src/course_work/training/engine.py` | Fix TRAIN_DIAGNOSTIC labeling |
| `src/course_work/training/engine.py` | Fix RMSE persistence (remove 0.0 fallbacks) |

---

## 3. Tests

### 3.1 Unit Tests

**A. FINAL_DEV Tests**
- TRAIN included in FINAL_DEV
- VALIDATION included in FINAL_DEV
- TEST excluded from FINAL_DEV
- Target IDs unique
- Combined count derived from source
- Chronological integrity
- Locked boundary protocol
- Locked feature order

**B. FINAL_SCALING Tests**
- X fit region = FINAL_DEV_REGION-v1
- Y fit region = FINAL_DEV_REGION-v1
- n_samples_seen consistent with FINAL_DEV
- No Test rows
- Same scaler reused across seeds
- Phase9 scalers not overwritten
- Roundtrip finite/pass
- SHA stable after freeze

**C. FINAL_REFIT Engine Tests**
- validation_loader=None
- evaluate_validation=False
- early stopping disabled
- Fixed epochs
- official_epoch = max_epochs
- No BEST-selection
- evaluate_validation=True behavior unchanged

**D. Driver Tests**
- Exactly seeds 42/123/2026
- Exactly 3 planned runs
- Fresh model per seed
- No Test accessor
- No score-based seed selection

**E. Release Tests**
- Cannot release with 0/3 seeds
- Cannot release with 1/3 seeds
- Cannot release with 2/3 seeds
- Cannot release with scaler mismatch
- Cannot release with population mismatch
- Cannot release with epoch mismatch
- Cannot release if Test accessed
- Release only when 3/3 verified

### 3.2 Integration Tests

- Phase 46 preflight passes all gates
- Test firewall verified
- FINAL_DEV + FINAL_SCALING pipeline end-to-end

---

## 4. Preflight Gates

All gates must pass before Phase 46 scientific rerun:

1. **Phase 45 PASS gate**: `phase_45_signoff.json` exists with `status=PASS`
2. **Handoff ready gate**: `phase46_three_seed_handoff.json` exists
3. **Lock hashes gate**: Final model lock SHA matches Phase 46 config SHA
4. **Canonical boundary gate**: Boundary protocol = WB0_CONTEXT_CARRY_OVER
5. **FINAL_DEV construction gate**: Build succeeds, TRAIN+VALIDATION included, TEST excluded
6. **TRAIN+VALIDATION population gate**: Population fingerprint matches Phase 45 lock
7. **Test exclusion gate**: Test count = 0 in FINAL_DEV
8. **FINAL_SCALING materialization gate**: Scalers fit on FINAL_DEV, not TEST
9. **Feature order gate**: Feature order matches Phase 45 lock
10. **Seed set gate**: Seeds = [42, 123, 2026]
11. **Epoch contract gate**: FINAL_REFIT_EPOCHS = locked value (expected 50)
12. **FINAL_REFIT engine semantics gate**: validate_validation=False, early_stopping=False
13. **Registry availability gate**: ExperimentRegistry accessible
14. **Test firewall gate**: No Test DataLoader, no Test target access

---

## 5. Scientific Rerun Boundary

**AFTER** all source corrections, tests, and preflight pass:

```
READY_FOR_PHASE46_SCIENTIFIC_RERUN = YES
```

The human user runs the terminal command manually.

**No agent automatic training.**

---

## 6. STOP Conditions

STOP and DO NOT proceed if any of the following occur:

1. Phase 45 lock artifacts not found or invalid
2. FINAL_DEV construction fails
3. FINAL_SCALING fit fails
4. Any unit test fails
5. Any integration test fails
6. Preflight gate fails
7. Test target access detected
8. Phase 47 execution detected
9. Phase 9 scaler files overwritten
10. Historical checkpoint bytes modified

---

## 7. Historical Artifact Preservation

Existing Phase 46 run IDs and checkpoints are preserved as:

```
HISTORICAL_INVALIDATED_BY_CONTRACT_DEVIATION
```

- `RUN_TR_FSD_0153_B15A19DC` (seed 42)
- `RUN_TR_FSD_0154_DD82D743` (seed 123)
- `RUN_TR_FSD_0155_59A50ADD` (seed 2026)

These are **NOT** modified, deleted, or overwritten.

New corrected Phase 46 runs will create **NEW** run IDs.

---

## 8. Test Access Rules

**HARD FORBIDDEN:**

- Construct Test DataLoader
- Materialize Test targets
- Run Test inference
- Run Test scaler transformation checks
- Compute Test metrics
- Start Phase 47

**ALLOWED:**

- TRAIN data access
- VALIDATION data access
- FINAL_DEV (TRAIN+VALIDATION) data access

---

## 9. Phase 47 Execution Rules

**HARD FORBIDDEN during this task:**

- Execute Phase 47
- Create Test predictions
- Compute Test metrics

**AFTER Phase 46 corrective rerun completes:**

- Phase 47 may execute only if `phase47_test_release.json` is written with `released=true`
- All gates must pass before Phase 47

---

## 10. Implementation Phases

### Phase A: Documentation (Step 1)
- [x] Issue document created
- [x] Corrective plan created

### Phase B: Preservation (Step 2)
- [x] Historical Phase 46 evidence tagged as preserved

### Phase C: FINAL_DEV Implementation (Step 3)
- [ ] `build_final_dev_dataset()` API in `src/course_work/data/final_dev.py`
- [ ] TRAIN + VALIDATION construction
- [ ] Population fingerprint
- [ ] Feature order verification
- [ ] Test exclusion verification

### Phase D: FINAL_SCALING Implementation (Step 4)
- [ ] `materialize_final_scaling_v1()` in `src/course_work/scaling/`
- [ ] Separate artifact path: `artifacts/scaling/final_dev/`
- [ ] X/Y scalers fit on FINAL_DEV_REGION-v1
- [ ] Artifact manifest, checksums, roundtrip tests
- [ ] Phase 9 scalers preserved (NOT overwritten)

### Phase E: Driver Wiring (Step 5)
- [ ] Wire `load_final_dev_x_scaler()` into Phase 46 driver
- [ ] Wire `load_final_dev_target_scaler()` into Phase 46 driver
- [ ] Add runtime SHA assertions
- [ ] Add fit_region assertions

### Phase F: FINAL_REFIT Semantics (Step 6)
- [ ] Fix `best_epoch` to equal `max_epochs` when `evaluate_validation=False`
- [ ] Guard `evaluate_validation=True` behavior unchanged

### Phase G: Metric Semantics (Step 7)
- [ ] Label metrics as `TRAIN_DIAGNOSTIC` when `evaluate_validation=False`
- [ ] Fix RMSE persistence (remove `0.0` fallbacks)
- [ ] Ensure consistency across all artifacts

### Phase H: Phase 47 Release (Step 8)
- [ ] Implement `build_phase47_release()` logic
- [ ] DO NOT create `released=true` artifact during source-only implementation
- [ ] Test release conditions (0/3, 1/3, 2/3, 3/3, mismatch cases)

### Phase I: Artifacts (Step 9)
- [ ] Complete generation logic for all O46.1-O46.41 outputs
- [ ] Follow actual project naming/location conventions

### Phase J: Checkpoint Contract (Step 10)
- [ ] Future checkpoints include all required provenance fields
- [ ] No Test metrics in checkpoints

### Phase K: Tests (Step 11)
- [ ] Unit tests for FINAL_DEV, FINAL_SCALING, FINAL_REFIT, Driver, Release
- [ ] Integration tests for Phase 46 preflight

### Phase L: Preflight (Step 12)
- [ ] `scripts/phase46_preflight.py` dry-run mode
- [ ] All 14 preflight gates verified

### Phase M: Test Firewall (Step 13)
- [ ] Verify no Test access
- [ ] Verify no Phase 47 execution

### Phase N: Test Suite (Step 14)
- [ ] Run focused tests
- [ ] Run relevant regression tests
- [ ] Report results

### Phase O: Dry Run (Step 15)
- [ ] Execute Phase 46 preflight in explicit DRY-RUN mode
- [ ] Prove FINAL_DEV, Test=0, Scalers, Seeds, Epochs, Firewall

### Phase P: Ready (Step 16)
- [ ] `READY_FOR_PHASE46_SCIENTIFIC_RERUN = YES/NO`

### Phase Q: Terminal Command (Step 17)
- [ ] Return exact terminal command for human user
- [ ] Include macOS caffeinate keep-awake command
- [ ] Include monitoring commands

---

## 12. Historical Checkpoint Archival (HISTORICAL_CHECKPOINT_ARCHIVAL-v1)

### 12.1 Scope

Before corrected Phase 46 scientific training is authorized, the historical
INVALIDATED checkpoints currently occupying the canonical
`artifacts/three_seed_final_runs/official_checkpoints/seed_*` paths MUST be
archived. They are preserved as scientific evidence but never referenced by
the corrected Phase 47 release gate.

### 12.2 Historical run IDs (evidence to be preserved)

| Seed | Historical run ID         | Path                                                                                |
|------|---------------------------|-------------------------------------------------------------------------------------|
| 42   | RUN_TR_FSD_0153_B15A19DC  | artifacts/three_seed_final_runs/official_checkpoints/seed_42/seed_42_FINAL_REFIT.pt  |
| 123  | RUN_TR_FSD_0154_DD82D743  | artifacts/three_seed_final_runs/official_checkpoints/seed_123/seed_123_FINAL_REFIT.pt|
| 2026 | RUN_TR_FSD_0155_59A50ADD  | artifacts/three_seed_final_runs/official_checkpoints/seed_2026/seed_2026_FINAL_REFIT.pt|

Status: `INVALIDATED_BY_CONTRACT_DEVIATION` (TRAIN-only bug, Phase 9 scaler
reuse, missing `phase47_test_release.json`, stale RMSE=0.0 metadata).
Documented in
`docs/plan-doc/analysis_error/phase_46_final_dev_scaling_implementation_mismatch_issue.md`.

### 12.3 Archive destination

```
artifacts/three_seed_final_runs/historical_checkpoints/
├── RUN_TR_FSD_0153_B15A19DC/
│   ├── seed_42_FINAL_REFIT.pt
│   └── seed_42_FINAL_REFIT_metadata.json
├── RUN_TR_FSD_0154_DD82D743/
│   ├── seed_123_FINAL_REFIT.pt
│   └── seed_123_FINAL_REFIT_metadata.json
└── RUN_TR_FSD_0155_59A50ADD/
    ├── seed_2026_FINAL_REFIT.pt
    └── seed_2026_FINAL_REFIT_metadata.json
```

Companion artifacts:
- `historical_checkpoints/historical_checkpoint_archive_manifest.json`
- `historical_checkpoints/historical_checkpoint_archive_audit.csv`

### 12.4 Archival procedure (safe by construction)

For EACH historical checkpoint:

1. Compute SHA256 of source `.pt` file (`source_sha`).
2. Read source metadata sidecar.
3. Copy (NOT move) `.pt` and metadata.json into the archive destination
   named after the historical run_id.
4. Compute SHA256 of archived `.pt` (`archived_sha`).
5. Require `source_sha == archived_sha == metadata.model_state_sha256`.
6. On mismatch: STOP — no source file is removed.

Only after ALL three checkpoints are checksum-verified in the archive,
the canonical active `.pt` paths in `official_checkpoints/seed_*/` may be
cleared (replaced with a placeholder metadata file indicating
`status = AWAITING_CORRECTED_PHASE46_RUN`, or removed entirely if the
project rules allow).

The original `metadata.json` files are also archived, but a fresh
placeholder metadata.json may be written at the canonical path to prevent
confusion between INVALIDATED historical and future CORRECTED checkpoints.

### 12.5 Archive manifest

`historical_checkpoint_archive_manifest.json` minimum fields:

```json
{
  "archive_version": "HISTORICAL_CHECKPOINT_ARCHIVAL-v1",
  "phase": 46,
  "historical_status": "INVALIDATED_BY_CONTRACT_DEVIATION",
  "runs": [
    {
      "seed": 42,
      "historical_run_id": "RUN_TR_FSD_0153_B15A19DC",
      "original_path": "artifacts/three_seed_final_runs/official_checkpoints/seed_42/seed_42_FINAL_REFIT.pt",
      "archived_path": "artifacts/three_seed_final_runs/historical_checkpoints/RUN_TR_FSD_0153_B15A19DC/seed_42_FINAL_REFIT.pt",
      "checkpoint_sha256": "<hex>",
      "metadata_path": "...",
      "archived_at": "<iso8601>"
    }
  ],
  "archive_verified": true,
  "test_access": false,
  "scientific_training": false,
  "status": "PASS"
}
```

### 12.6 Historical run directories

The ExperimentRegistry historical directories under
`artifacts/runs/RUN_TR_FSD_015{3,4,5}_*` are NEVER deleted. Their
config.json, status.json, training_history.csv, training.log, and
metrics are preserved verbatim.

Additive evidence (no destructive mutation) is written to:
- `artifacts/runs/RUN_TR_FSD_0153_B15A19DC/historical_phase46_invalidation_manifest.json`
- `artifacts/runs/RUN_TR_FSD_0154_DD82D743/historical_phase46_invalidation_manifest.json`
- `artifacts/runs/RUN_TR_FSD_0155_59A50ADD/historical_phase46_invalidation_manifest.json`

### 12.7 Phase 47 release exclusion

After future corrected Phase 46 scientific training, `phase47_test_release.json`
MUST:
- Reference only NEW corrected run IDs (e.g. RUN_TR_FSD_0156+).
- NEVER reference RUN_TR_FSD_0153, RUN_TR_FSD_0154, RUN_TR_FSD_0155.
- Include `historical_run_ids_excluded` field listing the three historical IDs.
- Reject release if any of those three IDs appear in `run_records[].run_id`.

### 12.8 Rollback condition

If ANY of the following occur, archival is ROLLED BACK and the historical
checkpoints remain at their canonical active paths:
- `source_sha != archived_sha`
- `metadata.model_state_sha256` does not match either source or archive
- metadata sidecar is missing required keys (`seed`, `run_id`, `phase`, `checkpoint_type`)
- A historical run ID is unintentionally mutated or deleted

### 12.9 No scientific mutation

This archival operation is a READ-MOSTLY data-preservation step:
- No model retraining.
- No test data access.
- No new official run IDs created.
- No new FINAL_REFIT checkpoints generated.
- Only safe byte-for-byte copies of existing evidence into a new path
  with checksum verification, followed by safe clearing of the canonical
  paths (placeholder metadata only, no fake corrected data).

---

## 11. Files Modified Summary

### New Files Created
- `docs/plan-doc/analysis_error/phase_46_final_dev_scaling_implementation_mismatch_issue.md`
- `docs/plan-doc/plan_before_process/phase_46_corrective_reimplementation_plan.md`
- `src/course_work/scaling/final_scaling.py`
- `src/course_work/data/final_dev.py`
- `src/course_work/training/final_refit.py`
- `tests/unit/test_final_dev.py`
- `tests/unit/test_final_scaling.py`
- `tests/integration/test_phase46_preflight.py`
- `scripts/phase46_preflight.py`

### Modified Files
- `scripts/phase46_three_seed_runs.py`
- `src/course_work/training/engine.py`

### NOT Modified
- Historical Phase 46 artifacts
- Phase 9 scaler files
- Phase 45 lock artifacts
- Any Test artifacts
