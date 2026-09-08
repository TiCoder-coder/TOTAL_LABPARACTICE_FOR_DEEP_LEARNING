# PHASE 0-5 NOTEBOOK IPYTHON RUNTIME FIX PLAN

## Plan metadata

```text
Plan ID: CW-PHASE-0005-N2-FIX-001
Issue ID: CW-PHASE-0005-N2-001
Scope: Notebook execution runtime directories only
Status: COMPLETED
```

## Implementation

```text
1. Verify the failed attempt left all notebook execution counts and outputs empty.
2. Create dedicated temporary IPython and Jupyter directories with restricted permissions.
3. Set IPYTHONDIR, JUPYTER_CONFIG_DIR, JUPYTER_DATA_DIR and JUPYTER_RUNTIME_DIR to those paths.
4. Set JUPYTER_PATH to the verified virtual-environment Jupyter data directory.
5. Preserve PYTHONPATH, PYTHONWARNINGS=error and the Phase 1 kernel name.
6. Execute the unchanged notebook once from the repository root.
7. Verify monotonic execution counts, zero error outputs and zero warning streams.
8. Revalidate Phase 0-5 checksums, tests and raw-data immutability.
```

## Acceptance criteria

```text
The correction changes no notebook source.
No runtime state is written to the user home directory.
The verified python3 kernel starts from the project virtual environment.
Every notebook code cell executes exactly once in order.
All Phase 0-5 sign-offs and checksums remain valid.
Raw data remains unchanged.
Phase 6 remains unstarted.
```

## Completion result

```text
IPython and Jupyter runtime state was redirected to dedicated temporary directories.
The unwritable user-home warning no longer occurs.
The notebook remained unexecuted and unchanged during this correction.
A separate managed-sandbox socket restriction is tracked independently.
```
