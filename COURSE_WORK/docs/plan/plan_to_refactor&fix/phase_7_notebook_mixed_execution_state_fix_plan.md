# Phase 7 Notebook Mixed Execution State Fix Plan

## Plan ID

`CW-PHASE-7-NOTEBOOK-STATE-FIX-001`

## Approval status

`APPROVED_BY_CURRENT_HUMAN_REQUEST`

## Steps

1. Load the structurally validated notebook in memory.
2. Execute every cell in order with a fresh project kernel and isolated writable runtime directories.
3. Reject execution on any error, traceback or warning stream.
4. Require strictly increasing unique execution counts and Phase 7 outputs.
5. Atomically replace the canonical notebook only after validation.
6. Rerun notebook boundary tests and the complete test suite.

## Rollback condition

If any cell fails, retain the current canonical notebook and diagnose that cell before retrying.
