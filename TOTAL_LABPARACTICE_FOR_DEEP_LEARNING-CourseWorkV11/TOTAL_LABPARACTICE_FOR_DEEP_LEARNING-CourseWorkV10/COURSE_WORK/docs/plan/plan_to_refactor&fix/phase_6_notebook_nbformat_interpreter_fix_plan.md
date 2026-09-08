# Phase 6 Notebook Nbformat Interpreter Fix Plan

## Plan ID

`CW-PHASE-6-NBFORMAT-FIX-001`

## Approval status

`APPROVED_BY_CURRENT_HUMAN_REQUEST`

## Steps

1. Verify the canonical notebook has not changed.
2. Execute the prepared structural edit with `venv/bin/python`.
3. Validate the notebook schema and required Phase 6 cell IDs.
4. Remove the temporary helper.
5. Continue notebook-boundary tests only after validation passes.

## Expected impact

Only the approved Phase 6 import, orchestration cells and Phase 0-6 boundary are added to `CourseWork.ipynb`.
