# PHASE 2 OFFICIAL ARCHIVE ACQUISITION BLOCKED ISSUE

## Issue metadata

```text
Issue ID: CW-PHASE-0005-P2-001
Parent plan: CW-REFACTOR-0005-001
Phase: 2
Status: RESOLVED
Severity: BLOCKING_PHASE_GATE
```

## Required acquisition

```text
Method: AQ0_direct_uci
Provider: UCI Machine Learning Repository
Dataset ID: 374
Expected archive: appliances+energy+prediction.zip
Expected member: energydata_complete.csv
```

## Observed result

The escalated network action was rejected before command execution.

```text
Official archive created: false
Partial download created: false
Canonical raw CSV overwritten: false
Phase 2 artifact created: false
Phase 2 sign-off created: false
Phase 3 started: false
```

## Integrity evidence

```text
Existing canonical CSV SHA-256:
2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d

Phase 0 sign-off: PASS
Phase 1 sign-off: PASS
Blocked-state integrity check: PASS
```

## Root cause

Phase 2 requires the official UCI archive to establish archive provenance and compare the extracted CSV against the protected existing CSV. Network permission was not granted, so that evidence cannot be produced locally.

Using the existing CSV alone would not satisfy the approved AQ0 contract. Using a mirror or re-exporting a DataFrame would violate the source contract.

## Required decision

Human must choose one of the approved recovery paths in `phase_2_official_archive_acquisition_recovery_plan.md`.

## Resolution

```text
Recovery path: A
Official archive downloaded: true
ZIP integrity: PASS
ZIP path safety: PASS
Official CSV checksum match: PASS
DATA-v1 sign-off: PASS
Phase 0 to Phase 2 regression: PASS
```
