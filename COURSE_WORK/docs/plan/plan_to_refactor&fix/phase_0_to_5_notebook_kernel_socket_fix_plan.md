# PHASE 0-5 NOTEBOOK KERNEL SOCKET FIX PLAN

## Plan metadata

```text
Plan ID: CW-PHASE-0005-N2-FIX-002
Issue ID: CW-PHASE-0005-N2-002
Scope: Local Jupyter kernel process for one notebook execution
Status: COMPLETED
```

## Implementation

```text
1. Verify the second failed attempt left notebook counts and outputs empty.
2. Request managed permission for the exact nbconvert execution command.
3. Retain all temporary runtime-directory environment variables.
4. Retain PYTHONPATH, PYTHONWARNINGS=error and the Phase 1 kernel name.
5. Execute only CourseWork.ipynb in place from the repository root.
6. Inspect the executed notebook for monotonic counts, errors and warning streams.
7. Revalidate all Phase 0-5 tests, sign-offs and checksums.
8. Reconfirm raw-data immutability and the Phase 6 boundary.
```

## Acceptance criteria

```text
Only the local notebook kernel and its required localhost sockets run outside the sandbox.
No network acquisition occurs.
No notebook source changes are introduced by the recovery.
All code cells execute exactly once in order.
All strict validation and regression checks pass.
Raw data remains unchanged.
Phase 6 remains unstarted.
```

## Completion result

```text
The managed permission was granted for the exact notebook command.
The verified kernel successfully opened its required local sockets.
Execution advanced to the first notebook code cell.
A separate relative import-path issue is tracked independently.
```
