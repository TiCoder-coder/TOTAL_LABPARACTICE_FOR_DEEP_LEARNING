# PHASE 1 NOTEBOOK WORKING DIRECTORY ISSUE

## Issue metadata

```text
Issue ID: CW-PHASE-0005-N2-004
Parent plan: CW-REFACTOR-0005-001
Step: N2
Status: RESOLVED
Severity: BLOCKING_CURRENT_STEP
```

## Observed result

The notebook passed imports and Phase 0 but Phase 1 rejected the current runtime as different from signed ENV-v1.

```text
Recorded working directory: repository root
Current kernel working directory: COURSE_WORK/notebook_course_work
Other environment fields differing: none
Kernel matches interpreter: true
Project root resolution: correct
Notebook overwritten by nbconvert: false
Phase 0-5 signed artifacts modified: false
```

## Root cause

Nbconvert assigns the input notebook directory as the kernel resource path. ENV-v1 was signed from the repository root. The Phase 1 contract records and validates the working directory so the mismatch is correctly rejected.

## Candidate directions

```text
Direction A: execute with an explicit kernel resource path equal to the signed repository root.
Direction B: remove or weaken working-directory equality in materialize_phase_1.
Direction C: change directory inside the notebook.
```

Direction A preserves ENV-v1, keeps source unchanged and enforces the contract. Direction B weakens a signed Phase 1 guard. Direction C violates the orchestration-only notebook boundary.

## Required correction

Use the standard nbclient executor with its resource metadata path set to the repository root, retain the verified kernel and runtime controls, write the notebook only after all cells complete, then run the complete validation chain.

## Resolution evidence

```text
Kernel resource path set to repository root: true
Recorded working directory equals current: true
Working-directory field repeated as drift: false
Remaining drift fields: mps_available, mps_device_name, selected_device
Notebook source changed by correction: false
```
