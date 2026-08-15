# PHASE 1 KERNEL RESOLUTION FIX PLAN

## Plan metadata

```text
Plan ID: CW-PHASE-0005-P1-FIX-001
Issue ID: CW-PHASE-0005-P1-001
Execution mode: SEQUENTIAL
Scope: Phase 1 kernel discovery only
Status: COMPLETED
```

## Objective

Correct virtual-environment kernelspec discovery without weakening interpreter equality or changing notebook metadata.

## Files allowed to change

```text
COURSE_WORK/src/course_work/utils/environment.py
COURSE_WORK/tests/unit/test_environment.py when additional regression coverage is required
```

## Implementation

```text
1. Preserve the unresolved active interpreter path.
2. Resolve it separately for executable identity comparison.
3. Search kernelspec under sys.prefix before user-level candidates.
4. Resolve a relative python kernel command against the active environment bin directory.
5. Compare resolved kernel and interpreter executables.
```

## Validation

```text
Kernel specification path is discovered.
Kernel executable matches the active interpreter.
All Phase 1 unit tests pass.
All Phase 0 contract tests still pass.
No Phase 1 artifact exists until the complete P1 test gate passes.
Raw SHA-256 remains unchanged.
```

## Stop condition

Stop Phase 1 if the corrected function still cannot prove kernel and interpreter identity.

## Completion result

```text
The active environment kernelspec is resolved through sys.prefix.
Resolved executable identity matches the active interpreter.
Phase 1 artifacts and sign-off were created only after the complete test gate passed.
```
