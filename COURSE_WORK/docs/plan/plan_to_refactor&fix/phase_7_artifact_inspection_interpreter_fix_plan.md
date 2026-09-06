# Phase 7 Artifact Inspection Interpreter Fix Plan

## Plan ID

`CW-PHASE-7-INSPECTION-INTERPRETER-FIX-001`

## Approval status

`APPROVED_BY_CURRENT_HUMAN_REQUEST`

## Steps

1. Preserve the completed Phase 7 artifacts without rewriting them.
2. Repeat the read-only inspection with `venv/bin/python`.
3. Validate variant, leakage and order CSV schemas and statuses.
4. Reinvoke `materialize_phase_7` to verify idempotent signed-artifact reload.
5. Continue only after all checks pass.
