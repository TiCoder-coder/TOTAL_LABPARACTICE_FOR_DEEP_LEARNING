# PHASE 0-5 NOTEBOOK KERNEL IMPORT PATH FIX PLAN

## Plan metadata

```text
Plan ID: CW-PHASE-0005-N2-FIX-003
Issue ID: CW-PHASE-0005-N2-003
Scope: Child-kernel source-path resolution only
Status: COMPLETED
```

## Implementation

```text
1. Verify the failed execution did not overwrite the clean notebook.
2. Resolve COURSE_WORK/src to its absolute repository path before kernel startup.
3. Pass that absolute path through PYTHONPATH without notebook path mutation.
4. Retain the verified kernel, strict warning policy and temporary runtime directories.
5. Execute the unchanged notebook once.
6. Verify every code cell executes exactly once and in order.
7. Verify no error or warning output is stored.
8. Revalidate all Phase 0-5 artifacts, checksums and tests.
```

## Acceptance criteria

```text
The source package imports without notebook bootstrap logic.
No packaging or dependency state changes.
No notebook source changes from the recovery.
All N2 final checks pass.
Raw data remains unchanged.
Phase 6 remains unstarted.
```

## Completion result

```text
The kernel imported course_work from the absolute approved source root.
No notebook bootstrap or sys.path mutation was added.
No package installation or dependency change occurred.
Execution reached the Phase 1 environment drift check.
```
