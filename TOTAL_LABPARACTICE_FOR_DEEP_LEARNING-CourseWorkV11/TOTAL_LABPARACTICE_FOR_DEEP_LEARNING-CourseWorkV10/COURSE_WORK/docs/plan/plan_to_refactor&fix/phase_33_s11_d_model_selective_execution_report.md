# Phase 33 S11 d_model Selective Execution Report

## 1. Scope

Phase 33 was implemented and executed as a terminal-owned selective sweep. The notebook was not executed or modified. Canonical artifacts, not processing logs, controlled reuse, execution and winner selection.

## 2. Implemented ownership

```text
Scientific contract: src/course_work/sweeps/d_model.py
Selective state: src/course_work/experiments/phase_execution.py
Terminal dispatch: scripts/run_single_condition.py
Dependency runner: scripts/run_all_pending.py
Canonical finalizer: src/course_work/sweeps/sweep_results.py
Derived presentation: src/course_work/reporting/phase_summary.py
Static processing log: docs/save_log_in_processing/phase_33_s11_d_model_log.json
Canonical root: artifacts/sweeps/S11_d_model
```

## 3. Frozen S11 contract

```text
D32: d_model = 32, TRAIN_NEW
D64: d_model = 64, REUSE_REFERENCE
Heads: 4
Layers: 2
FFN width: 128
Dropout: 0.1
Weight decay: 0.001
Learning rate: 0.0003
Batch size: 32
Selection metric: minimum full-precision Validation RMSE Wh
Exact tie rule: D32
Test access: FORBIDDEN
```

## 4. Sequential execution result

### 4.1 Preflight

```text
Phase 32 gate: PASS
Initial Phase 33 state: CONDITION_INCOMPLETE
Initial action: EXECUTE_MISSING_ONLY
Verified condition: D64
Missing condition: D32
D64 reference run: RUN_TR_S09_0016_AE0FB819
Capacity geometry audit: PASS
```

### 4.2 D32 training

```text
Run ID: RUN_TR_S11_0019_193AC913
Device: MPS
Completed epochs: 28
Best epoch: 18
Validation RMSE: 58.45725801944106 Wh
Trainable parameters: 26529
Head dimension: 8
FFN ratio: 4.0
```

Early stopping ended training after ten epochs without a better Validation RMSE. No Test metric was materialized.

### 4.3 Reused D64 evidence

```text
Run ID: RUN_TR_S09_0016_AE0FB819
Validation RMSE: 58.08190056355405 Wh
Trainable parameters: 69185
Head dimension: 16
FFN ratio: 2.0
```

### 4.4 Winner

```text
Winner condition: D64
Winner run: RUN_TR_S09_0016_AE0FB819
Winner Validation RMSE: 58.08190056355405 Wh
RMSE margin over D32: 0.37535745588701 Wh
Parameter difference D64 minus D32: 42656
Parameter increase relative to D32: 160.79007878171058 percent
Approved for Phase 34: true
```

The capacity context is descriptive. Winner selection used Validation RMSE only.

## 5. Canonical outputs

```text
artifacts/sweeps/S11_d_model/s11_d_model_metrics.csv
artifacts/sweeps/S11_d_model/s11_d_model_sweep_manifest.json
artifacts/sweeps/S11_d_model/s11_d_model_winner.json
artifacts/sweeps/S11_d_model/s11_reference_update.json
artifacts/sweeps/S11_d_model/phase_33_signoff.json
docs/save_log_in_processing/phase_33_s11_d_model_log.json
```

## 6. Verification

```text
Final phase state: VALID_REUSABLE
Final action: RENDER_ONLY
Sign-off validation: PASS
Condition completeness: D32 and D64 verified
Registry validation audit: 13 PASS, 0 FAIL
Test metric count: 0
Processing-log warnings: 0
Processing-log discrepancies: 0
HTML widget MIME: absent
HTML script tag: absent
Phase 32 canonical preservation: PASS
Notebook checksum preservation: PASS
Relevant regression tests: 86 PASS
Audit-only rerun: no required phases, no completed conditions, no scientific writes
```

The repository-wide suite also exposed unrelated historical and test-contract debt. It is isolated in `phase_33_full_repository_regression_scope_issue.md` and no protected artifact was changed to hide it.
