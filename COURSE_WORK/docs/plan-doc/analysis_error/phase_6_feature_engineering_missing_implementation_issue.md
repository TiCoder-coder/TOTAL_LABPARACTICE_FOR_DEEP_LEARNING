# Phase 6 Feature Engineering Missing Implementation Issue

## Issue ID

`CW-PHASE-6-MISSING-001`

## Status

`CONFIRMED`

## Current state

Phase 0 through Phase 5 are implemented and signed off. Phase 6 is specified by `Phase_6_Feature_engineering.md`, but its canonical owner `src/course_work/data/features.py` is empty, the Phase 6 artifact roots do not exist, the integration test still asserts that Phase 6 is unstarted, and `CourseWork.ipynb` stops after Phase 5.

## Required state

Phase 6 must produce `FEATURES-v1` from the signed `TEMPORAL-v1` view without mutating raw data or applying any fit-dependent preprocessing. The notebook must only call the Phase 6 public API and display saved outputs.

## Root cause

The repository was intentionally stopped at the Phase 0-5 boundary while the earlier refactor was validated. The Human has now explicitly authorized Phase 6.

## Impacted layers

- Phase contract: Phase 6 requirements and acceptance criteria.
- Source: deterministic feature engineering and artifact materialization.
- Data lifecycle: one derived CSV under `data/interim`.
- Artifacts: registry, lineage, availability, leakage audit, engineering audit, manifest, checksum, discrepancy log and sign-off.
- Tests: unit, integration and notebook-boundary coverage.
- Notebook: orchestration-only Phase 6 section.
- Architecture documentation: active scope and Phase-to-module mapping.

## Constraints

- Use only `src/course_work/data/features.py` as the Phase 6 implementation owner.
- Create exactly five engineered features: `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `weekend`.
- Use `timestamp_parsed` from `TEMPORAL-v1`; do not independently parse raw `date`.
- Preserve row order, timestamps, continuity segments, target, raw exogenous variables and random controls.
- Do not scale, impute, shift targets, create lags, create rolling model features, split data, build windows or train models.
- Do not place feature-processing logic in the notebook.
- Do not add dependencies, code comments or icons.

## Resolution gate

The issue is resolved only when Phase 6 artifacts reload correctly, all invariants pass, the derived checksum matches, Phase 0-6 integration passes, the notebook runs from a clean kernel, and the complete test suite passes.
