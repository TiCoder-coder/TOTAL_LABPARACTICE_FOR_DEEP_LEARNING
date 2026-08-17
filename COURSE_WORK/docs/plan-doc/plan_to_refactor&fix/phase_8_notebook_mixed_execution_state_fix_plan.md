# Phase 8 Notebook Mixed Execution State Fix Plan

## Plan ID

`CW-PHASE-8-NOTEBOOK-STATE-FIX-001`

## Approval status

`APPROVED_BY_CURRENT_HUMAN_REQUEST`

## Steps

1. Load the structurally validated notebook in memory.
2. Execute all Phase 0-8 cells with a fresh project kernel and isolated writable runtime directories.
3. Reject execution on any error, traceback or warning stream.
4. Require strictly increasing unique execution counts and Phase 8 outputs.
5. Atomically replace the canonical notebook only after complete validation.
6. Rerun notebook boundary and full repository suites.

## Rollback condition

If a cell fails, retain the current canonical notebook and diagnose the exact cell before retrying.
