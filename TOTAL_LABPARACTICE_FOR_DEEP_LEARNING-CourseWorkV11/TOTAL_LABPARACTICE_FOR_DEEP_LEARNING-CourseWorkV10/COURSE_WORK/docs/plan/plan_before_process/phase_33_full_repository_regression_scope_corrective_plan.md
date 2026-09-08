# Phase 33 Full Repository Regression Scope Corrective Plan

## 1. Plan identity

```text
Plan ID: CW-PHASE-33-FULL-REPOSITORY-REGRESSION-CORRECTIVE-001
Source issue: phase_33_full_repository_regression_scope_issue.md
Current status: PREPARED_NOT_EXECUTED
Phase 33 rerun required: false
Notebook mutation authorized: false
Historical artifact replacement authorized: false
```

## 2. Objective

Restore repository-wide regression-test consistency without changing the verified Phase 33 result, without retraining any complete condition and without silently rewriting historical scientific evidence.

## 3. Sequential correction flow

### Step 1. Freeze current evidence

Capture checksums for Phase 2, Phase 13-33 canonical artifacts, all processing logs and the notebook.

### Step 2. Separate runtime-dependent tests

Run GPU-dependent tests in an MPS-visible execution context. Tests must inspect an existing signed environment revision without rematerializing it merely to load upstream data.

### Step 3. Refactor stale registry assumptions

Replace fixed `run_count == 1` assertions with invariant checks that require the persistence baseline run to exist exactly once while allowing later approved runs.

### Step 4. Resolve mutable registry ownership

Remove mutable registry projections from immutable early-phase output checksum assertions or introduce a revisioned registry snapshot owned by the corresponding phase. No existing sign-off may be edited in place.

### Step 5. Resolve Phase 2 checksum drift

Identify when `README_SOURCE.md` and `dataset_manifest.json` changed. Restore from matching history or create a revisioned Phase 2 acquisition artifact set after explicit approval.

### Step 6. Refresh derived presentation logs

Regenerate only stale Phase 15-22 processing logs from currently validated canonical sources. Processing-log refresh must not rewrite scientific artifacts.

### Step 7. Resolve notebook execution-state policy

Choose one explicit policy:

```text
preserve mixed execution counts and update the test contract
clear every output and execution count
execute every notebook cell to a complete sequential state
```

The current Phase 33 authorization permits only the first option and does not permit notebook mutation.

### Step 8. Verify sequentially

After every approved correction, rerun the smallest affected test group before proceeding. Finish with the full suite in an MPS-visible context and verify Phase 32, Phase 33 and notebook preservation checksums again.

## 4. Stop conditions

Stop before any operation that would:

```text
change CourseWork.ipynb
rewrite a signed historical artifact in place
retrain a complete run
read Test metrics or targets
alter the Phase 33 winner or reference
```

Such an operation requires separate Human approval.
