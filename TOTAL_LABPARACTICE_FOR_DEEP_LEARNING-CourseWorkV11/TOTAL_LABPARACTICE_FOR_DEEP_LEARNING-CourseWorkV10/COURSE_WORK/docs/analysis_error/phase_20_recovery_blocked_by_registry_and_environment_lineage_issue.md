# Phase 20 Recovery Blocked by Registry and Environment Lineage Issue

## 1. Issue identity

```text
Issue ID: CW-PHASE-20-REGISTRY-ENVIRONMENT-LINEAGE-001
Detected during: CW-PHASE-33-UPSTREAM-RECOVERY-BEFORE-IMPLEMENTATION-v1 Step 2
Status: OPEN
Severity: BLOCKING
Verified date: 2026-08-22
Source mutation: NONE
Notebook mutation: NONE
Scientific execution: NOT_STARTED
```

## 2. Expected state

Phase 20 must reload from a checksum-valid Phase 19 sign-off, a compatible signed environment, a complete LSTM run and a complete signed Phase 20 output set.

The experiment registry must also validate all registered run identities, configs, artifact paths, checksums, metrics and Test-firewall rules.

## 3. Observed Phase 20 state

```text
State: CANONICAL_EVIDENCE_INVALID
Scientifically reusable: false
Run ID: RUN_LS_LS_0004_D26C1EC5
Registry status: COMPLETED
Validation RMSE: 60.22335322681946 Wh
```

Missing signed Phase 20 outputs:

```text
artifacts/lstm_baseline/lstm_baseline_run_summary.csv
artifacts/lstm_baseline/lstm_vs_persistence_validation.csv
artifacts/lstm_baseline/lstm_baseline_audit.csv
artifacts/runs/RUN_LS_LS_0004_D26C1EC5/checkpoints/best_checkpoint.pt
```

Missing required registry artifact:

```text
artifacts/runs/RUN_LS_LS_0004_D26C1EC5/training.log
```

The Phase 20 sign-off expects:

```text
artifacts/environment/environment_report.json
sha256 = 43f0bb3a1fb8d165b9c24c9e6101269320a4178383aceac420fe25d9709e8d54
```

The active canonical environment report is:

```text
artifacts/environment/environment_report.json
sha256 = 0a356d4fc47d82ca5f77a32a5854634f1eadaa193dddf947d526929950739ff5
environment_revision_id = ENV-R-20260822T092959850966Z
python_version = 3.10.11
selected_device = mps
```

The historical environment report matching the old Phase 20 checksum remains preserved at:

```text
artifacts/environment/_history/20260822T092959850966Z/environment_report.json
```

## 4. Deeper dependency findings

The first inspection scope started at Phase 20, but direct owner verification found earlier invalid dependencies:

```text
Phase 12: PASS
Phase 13: FAIL because live experiment registry validation fails
Phase 14: FAIL because persistence_validation_predictions.csv checksum mismatches
Phase 15: FAIL because environment_report.json input checksum mismatches
Phase 16: FAIL because environment_report.json input checksum mismatches
Phase 17: FAIL because environment_report.json input checksum mismatches
Phase 18: FAIL because environment_report.json input checksum mismatches
Phase 19: FAIL because environment_report.json input checksum mismatches
Phase 20: FAIL because environment input and signed outputs are invalid
```

Registry audit:

```text
Total registered runs: 12
Runs with missing registered files: 9
Missing registered files: 27
Affected completed families: LSTM_BASELINE, TRANSFORMER_BASELINE, S7_BATCH_SIZE
```

Every affected completed run is missing two required files and one optional prediction file. This makes `ExperimentRegistry.validate_registry()` fail and therefore invalidates the live Phase 13 contract.

## 5. Root cause

The workspace contains historical sign-offs and registry records whose declared artifacts were partially removed from their canonical run directories. A later environment recovery correctly preserved the old environment under `_history` and created a current-machine environment revision, but Phase 15 through Phase 20 still point to the former canonical environment path and checksum.

The existing Phase 20 public owner cannot repair this state safely:

1. it returns immediately to strict verification when `phase_20_signoff.json` exists;
2. strict verification fails on the environment input before output recovery;
3. its sign-off recovery path requires all summary files and a completed checkpoint already on disk;
4. fresh materialization would encounter existing write-once Phase 20 artifacts;
5. Phase 19 verification fails before a new run may start;
6. the live registry is globally invalid because registered artifact files are absent.

## 6. Why processing logs cannot resolve the issue

Files under `docs/save_log_in_processing` preserve presentation state only. They do not contain model state and cannot replace checkpoints, registry artifact files, signed histories or current-environment lineage.

## 7. Forbidden recovery shortcuts

```text
Do not fabricate a checkpoint from a metric JSON
Do not change a checksum to match a missing file
Do not treat a processing log as canonical evidence
Do not silently accept a historical environment as the active environment
Do not mutate a COMPLETED registry record in place
Do not overwrite surviving signed files
Do not continue to Phase 21 while Phase 20 is invalid
Do not access Test data
```

## 8. Required resolution

Execute a revisioned recovery from Phase 13 through Phase 21. Preserve the current invalid registry and canonical roots as historical evidence, create a clean current-environment registry revision, rematerialize non-training contracts, regenerate the Persistence baseline, and perform fresh current-environment LSTM and Transformer B0 runs from the terminal.

The required plan is:

```text
docs/plan-doc/plan_before_process/phase_13_to_21_revisioned_canonical_recovery_plan.md
```

## 9. Current gate

```text
Phase 20 recovery authorized: false
Phase 21 recovery authorized: false
Phase 22 recovery authorized: false
Phase 33 implementation authorized: false
Human approval required: true
```
