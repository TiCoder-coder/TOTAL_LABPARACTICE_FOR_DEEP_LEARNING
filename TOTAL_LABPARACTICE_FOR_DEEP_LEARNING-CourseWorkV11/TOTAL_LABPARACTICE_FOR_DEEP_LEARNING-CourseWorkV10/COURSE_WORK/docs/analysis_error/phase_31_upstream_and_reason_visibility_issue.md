# Phase 31 Upstream and Reason Visibility Issue

## 1. Issue identity

```text
Issue ID: CW-PHASE-31-UPSTREAM-GATE-001
Phase: 31
Detected at: 2026-08-22
Status: CONFIRMED
Severity: CRITICAL_FOR_SCIENTIFIC_EXECUTION
```

## 2. Observed state

Phase 31 requires a valid Phase 30 handoff containing:

```text
artifacts/sweeps/s8_learning_rate/phase_30_signoff.json
artifacts/sweeps/s8_learning_rate/s8_learning_rate_winner.json
artifacts/sweeps/s8_learning_rate/s8_reference_update.json
```

Current repository state:

```text
Phase 29 processing log: PASS
Phase 29 results.csv: MISSING
Phase 29 winner: MISSING
Phase 29 reference update: MISSING
Phase 30 processing log: BLOCKED
Phase 30 state: UPSTREAM_INVALID
Phase 30 winner: MISSING
Phase 30 reference update: MISSING
```

The Phase 29 and Phase 30 processing logs cannot replace the missing canonical scientific artifacts.

## 3. Additional resolver defect

The selective resolver correctly returns `UPSTREAM_INVALID` and `BLOCK`, but its top-level `reasons` list is empty when the resolved action is already `BLOCK`.

The HTML presentation therefore cannot provide a concise block reason even though prerequisite records contain exact missing paths.

## 4. Root causes

- Derived logs were previously treated as if they could prove scientific completion.
- Phase 29 and Phase 30 signed artifact sets are incomplete.
- Phase 31 was outside the active architecture scope.
- The generic resume planner only populated reasons while downgrading `EXECUTE_MISSING_ONLY` to `BLOCK`.

## 5. Correction direction

1. Extend architecture scope through Phase 31 after Human approval.
2. Add Phase 31 to the selective registry with exact WD0, WD1 and WD2 values.
3. Add a separate Phase 31 module that owns S9 contract and upstream preflight.
4. Propagate prerequisite and signoff failures into top-level resume reasons.
5. Generate a derived Phase 31 processing log and static HTML block report.
6. Add terminal-only Phase 31 dispatch without running upstream materializers.
7. Add one notebook public reporting call and no notebook processing logic.
8. Keep scientific execution blocked until Phase 30 canonical evidence becomes valid.

## 6. Must-not-fix shortcuts

- Do not change Phase 29 or Phase 30 log status to PASS manually.
- Do not create S8 winner or reference-update files from a log.
- Do not hard-code the selected learning rate.
- Do not pre-fill WD metrics or winner.
- Do not run WD0 or WD2 while Phase 30 is invalid.
- Do not retrain WD1.
- Do not access Test.
- Do not run the entire notebook.

## 7. Expected corrected behavior

```text
Phase 31 state: UPSTREAM_INVALID
Resolved action: BLOCK
Effective action: BLOCK
Execution authorized: false
Expected conditions: WD0, WD1, WD2
Block reason: exact missing or invalid Phase 30 canonical artifacts
```
