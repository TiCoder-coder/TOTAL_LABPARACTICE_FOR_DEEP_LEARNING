# PHASE 23 STALE SIGN-OFF RESUME PRECEDENCE ISSUE

## 1. Scope

The Phase 23 recovery audit found one verified reference condition and two unresolved conditions:

```text
FS0_TF1 missing
FS1_TF1 verified from the fresh Phase 21 baseline
FS2_TF1 missing
```

The active Phase 23 sign-off belongs to the preserved pre-recovery lineage and is invalid against the new canonical Phase 13 through Phase 22 chain.

## 2. Observed behavior

`inspect_phase_state` classified Phase 23 as `SIGNOFF_INVALID` before evaluating unresolved conditions. `plan_phase_resume` consequently returned `BLOCK`, and `run_all_pending.py` could not execute the two missing conditions.

## 3. Root cause

The state precedence placed an invalid derived sign-off before running, failed and missing condition states. A stale sign-off is derived evidence. It must not prevent recovery of unresolved scientific conditions when prerequisites and the registry remain valid.

## 4. Required behavior

State precedence must be:

```text
invalid prerequisite
running condition
failed or missing condition
invalid sign-off
missing derived artifact
missing or stale log
valid reusable
```

This preserves strict blocking for an invalid sign-off after all conditions are complete while allowing missing-only recovery before canonical finalization.

## 5. Safety constraints

```text
no condition already verified may run again
declared sign-off checksum failures remain blocking after condition completion
the stale sign-off may only be replaced by canonical finalization
Test access remains forbidden
accelerator readiness remains mandatory before training
the notebook remains unchanged
```

