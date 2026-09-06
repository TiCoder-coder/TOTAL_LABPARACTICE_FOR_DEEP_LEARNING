# Phase 6 Zero-Test Discovery Issue

## Issue ID

`CW-PHASE-6-TEST-DISCOVERY-001`

## Status

`CONFIRMED`

## Failure

The first full-suite `unittest discover` command exited successfully but collected zero tests. This result is not valid verification evidence.

## Root cause

The repository test layout and the selected discovery root were not resolved as an importable unittest hierarchy by that command.

## Impact

No source, artifact or notebook file was changed. Targeted unit, integration and notebook suites had already passed, but full-suite verification remained incomplete.

## Required correction

Use the project environment's `pytest` runner, verify that collection is nonzero, and require all collected tests to pass.
