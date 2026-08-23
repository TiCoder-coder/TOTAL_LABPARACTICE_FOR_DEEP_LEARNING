# Phase 33 Upstream Recovery Before Implementation Plan

## 1. Plan identity

```text
Plan ID: CW-PHASE-33-UPSTREAM-RECOVERY-BEFORE-IMPLEMENTATION-v1
Target phase: Phase 33 S11 d_model Sweep
Current status: COMPLETED
Scientific execution status: COMPLETED_VALIDATION_ONLY
Source implementation status: PHASE_33_IMPLEMENTED_AND_VERIFIED
```

## 2. Objective

Restore the canonical dependency chain required by Phase 33 without rerunning the notebook, without treating processing logs as scientific evidence and without executing any downstream condition before its immediate upstream handoff is valid.

The existing Phase 33 implementation plan resumes only after Phase 32 has a checksum-valid winner, reference update and sign-off with `approved_for_phase33 = true`.

## 3. Verified current state

```text
Phase 22 scientifically reusable = false
Earliest invalid canonical phase = Phase 20
Required recovery chain = Phase 20 through Phase 30
Phase 31 state = UPSTREAM_INVALID
Phase 32 state = UPSTREAM_INVALID
Phase 32 missing conditions = DR01, DR02, DR03
Phase 32 scientific artifact root = absent
Phase 32 derived processing log = BLOCKED
Phase 33 scientific execution = forbidden
```

## 4. Authority boundary

Canonical evidence may come only from:

```text
experiment registry
signed phase artifacts
run configs
checkpoints
histories
Validation metrics
winner artifacts
reference updates
declared checksums
```

Files under `docs/save_log_in_processing` may restore presentation state only. They cannot authorize training, provide a winner or replace missing canonical evidence.

## 5. Sequential recovery flow

### Step 1. Capture a recovery preservation baseline

Record checksums for:

```text
experiment registry
Phase 20 through Phase 32 canonical roots
Phase 20 through Phase 32 processing logs
notebook cell IDs, sources, outputs and execution counts
current environment revision
```

Verification:

```text
baseline capture is read-only
no scientific root is created
no notebook output changes
```

### Step 2. Audit Phase 20 and Phase 21 independently

Resolve the exact missing or invalid learned-baseline evidence for LSTM and Transformer B0.

Verification:

```text
no Test access
no silent artifact replacement
only invalid or missing learned-baseline work is scheduled
```

Stop and create a phase-specific corrective plan if either baseline cannot be safely recovered through existing public owners and terminal entry points.

Verified execution result on 2026-08-22:

```text
Phase 20 cannot be safely recovered through its current public owner
Phase 20 sign-off input environment checksum is stale
Phase 20 signed output set is incomplete
Phase 19 is not reloadable under the current environment revision
Phase 18 is not reloadable under the current environment revision
Phase 15 through Phase 17 are not reloadable under the current environment revision
Phase 14 has an output checksum mismatch
Phase 13 live registry validation fails
9 of 12 registry runs have missing physical artifact files
27 registered artifact files are missing
```

Execution is paused before Phase 20 recovery. The corrective plan is:

```text
docs/plan-doc/plan_before_process/phase_13_to_21_revisioned_canonical_recovery_plan.md
```

### Step 3. Recover Phase 22

Run the Phase 22 recovery only after Phase 20 and Phase 21 are canonical.

Verification:

```text
both baseline inputs reload with valid checksums
learning diagnostics are derived from canonical histories
Phase 22 sign-off is valid
```

### Step 4. Recover Phase 23 through Phase 30 sequentially

Use the existing missing-only terminal runner with dependencies enabled.

For each phase:

```text
inspect state
execute only missing authorized conditions
verify each completed condition immediately
finalize only complete verified evidence
refresh only that phase processing log
reinspect state before continuing
```

No phase may be skipped and no processing log may substitute for a canonical handoff.

### Step 5. Execute and finalize Phase 31

After Phase 30 is canonical:

```text
reuse exact WD1 reference
run only missing WD0 and WD2 conditions
verify Validation-only evidence
finalize Phase 31
verify approved_for_phase32 = true
```

### Step 6. Execute and finalize Phase 32

After Phase 31 is canonical:

```text
reuse exact DR01 reference
run only missing DR02 and DR03 conditions
verify the eight-site dropout scope
verify deterministic evaluation behavior
finalize Phase 32
verify approved_for_phase33 = true
```

### Step 7. Re-audit the Phase 33 gate

Required PASS conditions:

```text
s10_dropout_winner.json present and valid
s10_reference_update.json present and valid
phase_32_signoff.json present and valid
approved_for_phase33 = true
winner run config and registry evidence valid
Test firewall confirmed
current environment compatible
```

### Step 8. Resume Phase 33 implementation

Resume `phase_33_s11_d_model_selective_execution_plan.md` from its architecture and preservation steps only after Step 7 passes.

### Step 9. Execute Phase 33 from terminal

After the Phase 33 implementation and focused tests pass:

```text
reuse exact D64 reference
run only fresh D32
verify D32 immediately
finalize Phase 33
refresh only Phase 33 derived log
render Phase 33 from its notebook presentation cell
```

## 6. Immediate stop conditions

```text
active environment is incompatible with the signed environment
Phase 20 or Phase 21 cannot be recovered by existing public owners
any canonical checksum is invalid without an approved revision path
Test access is detected
a runner attempts to materialize an unapproved upstream phase
a condition would be retrained despite valid reusable evidence
an unrelated notebook cell or stored output changes
an existing valid signed artifact would be overwritten
```

## 7. Expected mutations after approval

The exact mutation set depends on the read-only audits. Permitted mutations are limited to invalid or missing canonical outputs in the approved recovery chain, derived logs for successfully verified phases and the later Phase 33 implementation files.

No valid upstream evidence may be overwritten in place.

## 8. Verification strategy

```text
immediate phase-state inspection after every step
registry and artifact checksum validation
Validation-only metric verification
Test-firewall audit
selective-runner audit and dry-run
notebook preservation hash comparison
focused unit and contract tests
related regression tests
final Phase 32 to Phase 33 handoff reload
```

## 9. Approval gate

```text
Issue documented: true
Corrective plan created: true
Source code changed in this turn: false
Notebook changed in this turn: false
Scientific execution started: false
Preservation baseline verified: true
Execution reached: Step 2
Waiting for corrective-plan approval: true
```
