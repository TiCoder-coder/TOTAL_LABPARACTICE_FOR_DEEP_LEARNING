# PHASE 3 NONNUMERIC FIXTURE DTYPE FIX PLAN

## Plan metadata

```text
Plan ID: CW-PHASE-0005-P3-FIX-001
Issue ID: CW-PHASE-0005-P3-001
Scope: One Phase 3 failure-path fixture
Status: COMPLETED
```

## Implementation

```text
1. Keep the canonical DataFrame untouched.
2. Deep-copy the fixture DataFrame.
3. Cast the copied Appliances Series to object.
4. Insert the invalid token into the copy.
5. Run all Phase 3 tests with warnings promoted to errors.
6. Run P0-P2 regression before Phase 3 materialization.
```

## Acceptance criteria

```text
No warning is emitted by fixture construction.
The numeric-coercion failure path remains tested.
All Phase 3 tests pass.
No Phase 3 artifact exists before the full gate passes.
Raw SHA-256 remains unchanged.
```
