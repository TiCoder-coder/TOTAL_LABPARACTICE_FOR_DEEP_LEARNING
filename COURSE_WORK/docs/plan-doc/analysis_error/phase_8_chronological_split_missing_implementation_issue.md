# Phase 8 Chronological Split Missing Implementation Issue

## Issue ID

`CW-PHASE-8-MISSING-001`

## Status

`CONFIRMED`

## Current state

Phase 7 is signed off as `FEATURESETS-v1`, but the canonical Phase 8 owner `src/course_work/data/splitting.py` is empty, `artifacts/splits` does not exist, the integration boundary asserts that Phase 8 is unstarted, and the notebook stops at Phase 7.

## Required state

Phase 8 must deterministically assign every ordered `FEATURES-v1` row to `TRAIN`, `VALIDATION` or `TEST` using the fixed 70/15/15 observation proportions and floor-based cumulative boundaries. The result must be `SPLIT-v1`, with target-timestamp assignment semantics, WB0 primary metadata, WB1 downstream support and an active Test firewall.

## Canonical owner

```text
src/course_work/data/splitting.py
```

No additional Phase 8 Python source file is needed.

## Runtime boundary evidence

For 19,735 signed rows:

```text
TRAIN rows = 13,814
VALIDATION rows = 2,960
TEST rows = 2,961
```

The first Validation timestamp is `2016-04-16 15:20:00` and the first Test timestamp is `2016-05-07 04:40:00`. Both boundaries lie within `SEG-0001`; they must not be shifted.

## Constraints

- No shuffle, random split or seed-dependent logic.
- No scaling, imputation, feature filtering, windows, loaders or training.
- No duplicated Train/Validation/Test feature CSV files.
- No Test target or feature distribution diagnostics before Phase 47.
- Train/Validation descriptive diagnostics must not alter boundaries.
- Every feature variant and model must share one split membership.
- Notebook must only call the Phase 8 public API and display saved outputs.
- Do not add dependencies, code comments or icons.

## Resolution gate

The issue is resolved only when all rows have exactly one chronological membership, boundaries and fingerprints reload identically, Test remains structurally locked, Phase 0-8 integration passes, the notebook executes cleanly, and the full test suite passes.
