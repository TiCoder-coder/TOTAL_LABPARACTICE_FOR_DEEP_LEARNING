# Phase 6 Zero-Test Discovery Fix Plan

## Plan ID

`CW-PHASE-6-TEST-DISCOVERY-FIX-001`

## Approval status

`APPROVED_BY_CURRENT_HUMAN_REQUEST`

## Steps

1. Collect tests with `venv/bin/python -m pytest COURSE_WORK/tests --collect-only`.
2. Fail verification if zero tests are collected.
3. Run the complete collected suite.
4. Preserve the targeted suite results as secondary evidence.
5. Continue final integrity checks only after the complete suite passes.
