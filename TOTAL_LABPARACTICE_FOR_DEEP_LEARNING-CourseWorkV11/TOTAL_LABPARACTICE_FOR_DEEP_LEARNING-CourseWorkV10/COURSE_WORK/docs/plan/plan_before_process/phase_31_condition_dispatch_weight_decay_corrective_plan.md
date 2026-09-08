# PHASE 31 CONDITION DISPATCH WEIGHT-DECAY CORRECTIVE PLAN

## 1. Objective

Correct condition-specific configuration ownership without changing any scientific contract.

## 2. Sequence

### Step 1

Introduce one pure configuration resolver for prepared Phase 31 and Phase 32 condition payloads.

### Step 2

Resolve the swept value from the condition payload and resolve all frozen values from the handoff.

### Step 3

Add regression tests for Phase 31 payloads without frozen `weight_decay` and Phase 32 payloads with selected frozen weight decay.

### Step 4

Run focused dispatcher, weight-decay, dropout, phase-execution and runner tests.

### Step 5

Reinspect Phase 31 and retry only `WD0` and `WD2` after accelerator readiness passes.

## 3. Stop conditions

```text
verified WD1 is selected for rerun
a failed Phase 31 record exists
upstream Phase 30 becomes invalid
MPS or CUDA is unavailable
Test access is requested
```

## 4. Authorization

This correction is required to continue the approved sequential Phase 31 recovery and does not expand its scope.

