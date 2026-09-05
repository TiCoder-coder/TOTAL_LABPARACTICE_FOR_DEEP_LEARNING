# Phase 6 Notebook Mixed Execution State Fix Plan

## Plan ID

`CW-PHASE-6-NOTEBOOK-STATE-FIX-001`

## Approval status

`APPROVED_BY_CURRENT_HUMAN_REQUEST`

## Steps

1. Read the structurally validated notebook into memory.
2. Execute every cell in order with a fresh kernel and isolated writable runtime directories.
3. Reject the execution if any cell emits an error, traceback or warning.
4. Validate increasing unique execution counts and the Phase 6 outputs.
5. Atomically replace the canonical notebook.
6. Rerun the notebook boundary suite and the full repository suite.

## Rollback condition

If execution fails, retain the original canonical notebook and diagnose the exact failing cell before any retry.
