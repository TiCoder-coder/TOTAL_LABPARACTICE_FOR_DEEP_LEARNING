# PHASE 1 NOTEBOOK WORKING DIRECTORY FIX PLAN

## Plan metadata

```text
Plan ID: CW-PHASE-0005-N2-FIX-004
Issue ID: CW-PHASE-0005-N2-004
Scope: Notebook executor resource path only
Status: COMPLETED
```

## Implementation

```text
1. Verify the failed nbconvert attempt left the source notebook clean.
2. Load the notebook through the installed nbformat package.
3. Configure NotebookClient with the verified python3 kernel and repository-root resource path.
4. Retain absolute PYTHONPATH, strict warnings and temporary runtime directories.
5. Execute all notebook cells in memory.
6. Write to a sibling temporary notebook only after complete execution.
7. Atomically replace CourseWork.ipynb with the completed notebook.
8. Run notebook-state, artifact, checksum, test and Phase 6 boundary verification.
```

## Acceptance criteria

```text
Phase 1 observes the signed repository working directory.
ENV-v1 requires no weakening or regeneration.
Notebook source remains orchestration-only.
Failed execution cannot partially overwrite the canonical notebook.
All N2 validation checks pass.
Raw data remains unchanged.
Phase 6 remains unstarted.
```

## Completion result

```text
The kernel observed the repository root required by ENV-v1.
No working-directory validation was weakened.
No notebook directory mutation was added.
A separate managed-boundary MPS visibility drift is tracked independently.
```
