# Phase 13 to Phase 21 Revisioned Canonical Recovery Plan

## 1. Plan identity

```text
Plan ID: CW-PHASE-13-21-REVISIONED-CANONICAL-RECOVERY-v1
Parent plan: CW-PHASE-33-UPSTREAM-RECOVERY-BEFORE-IMPLEMENTATION-v1
Scope: Phase 13 through Phase 21
Status: WAITING_FOR_HUMAN_APPROVAL
Scientific execution status: NOT_STARTED
Notebook execution required: false
```

## 2. Objective

Restore a checksum-valid current-environment canonical chain from the experiment registry through both learned baselines without rerunning the notebook and without accepting missing historical run artifacts as current scientific evidence.

The recovery must preserve all current evidence, use public phase owners, keep Test locked, verify every phase immediately and stop at the first failed gate.

## 3. Recovery boundary

The approved upstream recovery originally began at Phase 20. Direct verification changed the earliest actionable boundary:

```text
Phase 12 is valid
Phase 13 live registry is invalid
Phase 14 output evidence is invalid
Phase 15 through Phase 19 are stale against the current environment revision
Phase 20 and Phase 21 baseline evidence is incomplete
```

This plan does not change Phase 0 through Phase 12, raw data, processed data, split membership, scalers, windows, metrics, model definitions, notebook outputs or the Test firewall.

## 4. Recovery strategy

Use a revisioned rebuild instead of altering completed historical records in place.

The existing Phase 13 through Phase 21 canonical roots and run registry are first preserved under a recovery history namespace. The active canonical locations are then rebuilt through the existing phase owners under the current signed environment.

Historical evidence remains readable and recoverable. New sign-offs may reference only newly verified current canonical paths.

## 5. Sequential execution flow

### Step 1. Capture the Phase 13 through Phase 21 preservation manifest

Record path, size and checksum for:

```text
artifacts/experiments
artifacts/runs
artifacts/baselines/persistence
artifacts/models/lstm
artifacts/models/transformer
artifacts/attention_verification
artifacts/forward_sanity
artifacts/training_engine
artifacts/lstm_baseline
artifacts/transformer_b0
artifacts/environment
notebook_course_work/CourseWork.ipynb
```

Verification:

```text
manifest is complete
notebook hash matches the Phase 33 preservation baseline
no scientific file changes
```

### Step 2. Add a terminal-owned revision recovery entry point

Create a recovery orchestrator under `scripts` that:

```text
accepts audit-only and execute modes
uses the project interpreter explicitly
creates one unique recovery revision ID
preserves affected canonical roots before activation
dispatches existing public phase owners in order
records every gate result
stops immediately on failure
never imports notebook code
```

No data processing or training logic may be duplicated in the script.

Verification:

```text
audit-only performs no mutation
unknown arguments are rejected
Test access is not requested
source compiles
focused tests pass
```

### Step 3. Preserve the invalid active evidence

Move the affected canonical roots into a unique recovery history directory without deleting files:

```text
artifacts/_history/<recovery_revision_id>/experiments
artifacts/_history/<recovery_revision_id>/runs
artifacts/_history/<recovery_revision_id>/baselines/persistence
artifacts/_history/<recovery_revision_id>/models/lstm
artifacts/_history/<recovery_revision_id>/models/transformer
artifacts/_history/<recovery_revision_id>/attention_verification
artifacts/_history/<recovery_revision_id>/forward_sanity
artifacts/_history/<recovery_revision_id>/training_engine
artifacts/_history/<recovery_revision_id>/lstm_baseline
artifacts/_history/<recovery_revision_id>/transformer_b0
```

Write a manifest that maps every old canonical path to its historical path and preserves its original checksum.

Verification:

```text
every original file exists at exactly one preserved historical path
all copied or moved bytes retain their checksums
no raw, interim, processed or Test data moves
current environment revision remains unchanged
operation is reversible from the manifest
```

### Step 4. Rematerialize Phase 13

Run the existing Phase 13 owner against valid Phase 12 evidence to create a clean active experiment registry.

Verification:

```text
Phase 13 sign-off PASS
registry validation PASS
active registry starts without historical incomplete runs
historical registry remains unchanged in recovery history
Test firewall PASS
```

### Step 5. Rematerialize Phase 14

Run the existing Persistence owner on the locked Validation population.

Verification:

```text
prediction count matches Validation population
MAE, RMSE and R2 use METRICS-v1
all Phase 14 output checksums reload
Test remains locked
```

### Step 6. Rematerialize Phase 15 through Phase 19 sequentially

Execute the existing public owners in this exact order:

```text
Phase 15 LSTM implementation
Phase 16 Transformer implementation
Phase 17 attention-aware encoder verification
Phase 18 forward-pass sanity
Phase 19 baseline training engine
```

After each phase:

```text
reload its sign-off
verify every declared input checksum
verify every declared output checksum
verify current environment lineage
verify next-phase handoff
stop before the next phase on any failure
```

### Step 7. Execute a fresh Phase 20 LSTM baseline

Use the existing Phase 20 owner and the current environment revision. The run must use the frozen LSTM_B0 configuration and Validation only.

Verification:

```text
registry status COMPLETED
required config, status, log, checkpoint and metric files present
best checkpoint checksum registered
Phase 20 summary, comparison and audit files present
Phase 20 sign-off reloads
no Test metrics or predictions exist
```

### Step 8. Execute a fresh Phase 21 Transformer B0 baseline

Use the existing Phase 21 owner only after Phase 20 passes.

Verification:

```text
registry status COMPLETED
required config, status, log, checkpoint and metric files present
Transformer B0 comparison uses the new Phase 20 run
attention-aware model contract remains valid
Phase 21 sign-off reloads
no Test access occurs
```

### Step 9. Reinspect Phase 22 recovery eligibility

Run the read-only recovery inspector.

Required result:

```text
Phase 20 scientifically reusable = true
Phase 21 scientifically reusable = true
Phase 22 required action = recover diagnostics or render existing valid diagnostics
earliest invalid phase is not earlier than Phase 22
```

Only then may the parent plan resume at Phase 22.

### Step 10. Refresh presentation logs without changing the notebook

Refresh Phase 13 through Phase 21 derived logs only after their canonical sign-offs pass.

Verification:

```text
logs derive from canonical JSON and CSV artifacts
HTML is static
no widget MIME is introduced
notebook source, outputs and execution counts remain unchanged
```

## 6. Immediate stop conditions

```text
preservation checksum mismatch
history destination already exists
Phase 12 becomes invalid
current environment loses MPS or CUDA readiness
any phase owner accesses Test
any public owner attempts to overwrite an unpreserved file
registry validation fails after Phase 13
any sign-off cannot reload immediately
Phase 20 or Phase 21 run is not COMPLETED
notebook hash changes
```

## 7. Expected mutations after approval

```text
one recovery history tree under artifacts/_history
one recovery preservation manifest
one terminal recovery orchestrator
focused recovery tests
new active canonical Phase 13 through Phase 21 artifacts
new Phase 20 and Phase 21 run artifacts
new active registry records for current-environment runs
refreshed Phase 13 through Phase 21 derived logs
one execution report after completion
```

No notebook cell, stored output, raw data or Test result is permitted to change.

## 8. Verification commands and evidence

Verification must cover:

```text
Python compilation
focused unit tests
registry validation
phase-owner sign-off reloads
artifact path and checksum validation
Validation population identity
Test-firewall audit
recovery-history manifest audit
notebook preservation hash comparison
git diff whitespace check
```

Each completed step must be appended to a recovery execution report under:

```text
docs/plan-doc/plan_to_refactor&fix
```

## 9. Approval gate

```text
Issue documented: true
Corrective plan created: true
Source changed for this corrective plan: false
Artifacts relocated: false
Training started: false
Notebook changed: false
Waiting for Human approval: true
```

## 10. Approved execution status

```text
Human approval received: true
Phase 13 through Phase 21 canonical recovery: complete
Phase 22 diagnostics recovery: complete
Phase 23 selective-resume audit: in progress
Phase 23 through Phase 30 training: not started
Notebook changed: false
```

The Phase 23 audit discovered stale sign-off precedence blocking missing-only recovery. The corrective insertion is defined in:

```text
docs/plan-doc/analysis_error/phase_23_stale_signoff_resume_precedence_issue.md
docs/plan-doc/plan_before_process/phase_23_stale_signoff_resume_precedence_corrective_plan.md
```
