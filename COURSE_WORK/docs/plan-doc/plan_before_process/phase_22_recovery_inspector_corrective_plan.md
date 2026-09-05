# Phase 22 Recovery Inspector Corrective Plan

## 1. Plan identity

```text
Plan ID: CW-PHASE-22-RECOVERY-INSPECTOR-CORRECTIVE-v1
Parent plan: CW-PHASE-33-UPSTREAM-RECOVERY-BEFORE-IMPLEMENTATION-v1
Authorization: APPROVED_BY_PARENT_RECOVERY_SCOPE
Status: IN_PROGRESS
```

## 2. Objective

Remove false baseline-invalid classifications and permit non-training Phase 22 recovery without weakening checksum validation, Test protection or accelerator requirements for training phases.

## 3. Sequential flow

### Step 1. Preserve current canonical evidence

Reuse the verified Phase 13 through Phase 21 recovery history, preservation manifest and notebook hash.

### Step 2. Correct output validation

Require every declared output path to exist. Validate its checksum whenever one is declared. Do not require a recursive checksum for a sign-off itself or a checksum for mutable live-registry outputs.

### Step 3. Correct recovery ordering

Move the Phase 22 recovery block before the accelerator gate. Apply the accelerator gate immediately after Phase 22 and before Phase 23 execution.

### Step 4. Add focused tests

Verify:

```text
unsigned existing output is accepted
declared checksum mismatch is rejected
invalid baseline remains blocked
Phase 22 recovery precedes accelerator blocking
audit-only remains read-only
```

### Step 5. Reinspect and recover Phase 22

Require both fresh baseline sources to be scientifically reusable, recover Phase 22 from canonical histories and reload its sign-off.

## 4. Stop conditions

```text
missing baseline history
declared checksum mismatch
Test access
notebook mutation
accelerator gate bypass for Phase 23 or later
```
