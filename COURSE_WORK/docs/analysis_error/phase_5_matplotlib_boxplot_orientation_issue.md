# PHASE 5 MATPLOTLIB BOXPLOT ORIENTATION ISSUE

## Issue metadata

```text
Issue ID: CW-PHASE-0005-P5-001
Parent plan: CW-REFACTOR-0005-001
Phase: 5
Status: RESOLVED
Severity: BLOCKING_CURRENT_PHASE
```

## Observed result

Matplotlib raised `PendingDeprecationWarning` for the `vert` boxplot argument while warnings were promoted to errors.

```text
Computation tests: 5/5 PASS
Failure location: EDA-03 rendering
EDA manifest created: false
Phase 5 sign-off created: false
Phase 6 started: false
```

Tables and the first completed figures are unsigned partial artifacts. They must be byte-verified during recovery and cannot be treated as EDA-v1 evidence before the final sign-off.

## Root cause

The implementation used the legacy `vert=False` boxplot argument supported by the current Matplotlib release but scheduled for deprecation. The approved strict validation policy rejects unresolved warnings.

## Required correction

Use the current `orientation="horizontal"` API for target, temperature and humidity boxplots, then rerun materialization with warnings promoted to errors.

## Resolution evidence

```text
Correction: all three legacy arguments replaced with orientation="horizontal"
Strict Phase 5 tests: 5/5 PASS
Full Phase 0-5 test suite: 37/37 PASS
Canonical tables: 16/16 verified
Canonical figures: 16/16 verified
Visual inspection: PASS
Phase 5 sign-off: PASS
Raw SHA-256: 2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d
Phase 6 started: false
```
