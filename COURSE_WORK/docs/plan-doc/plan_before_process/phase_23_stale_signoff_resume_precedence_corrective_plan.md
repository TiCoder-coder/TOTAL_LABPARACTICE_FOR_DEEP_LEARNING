# PHASE 23 STALE SIGN-OFF RESUME PRECEDENCE CORRECTIVE PLAN

## 1. Objective

Correct Phase 23 selective resume so unresolved conditions can execute without weakening sign-off validation or rerunning verified evidence.

## 2. Authorized sequence

### Step 1

Add a focused regression test proving that an invalid stale sign-off plus unresolved conditions resolves to `CONDITION_INCOMPLETE` and `EXECUTE_MISSING_ONLY`.

### Step 2

Move invalid sign-off classification after running, failed and missing condition classification in `inspect_phase_state`.

### Step 3

Verify existing complete-condition sign-off failure tests still resolve to `SIGNOFF_INVALID` and `BLOCK`.

### Step 4

Verify `run_all_pending.py` selects only `FS0_TF1` and `FS2_TF1` for Phase 23 when execution becomes available.

### Step 5

Reinspect Phase 23 and the Phase 23 through Phase 30 chain without writing scientific artifacts.

## 3. Execution gate

Training may start only when the current runtime exposes CUDA or MPS. A CPU fallback is not authorized.

## 4. Expected result

```text
Phase 23 state CONDITION_INCOMPLETE
Phase 23 resolved action EXECUTE_MISSING_ONLY
FS1_TF1 remains reusable
FS0_TF1 and FS2_TF1 remain the only execution targets
invalid complete sign-offs remain blocked
```

## 5. Approval

This corrective plan is a necessary implementation detail within the approved Phase 13 through Phase 30 canonical recovery scope.

