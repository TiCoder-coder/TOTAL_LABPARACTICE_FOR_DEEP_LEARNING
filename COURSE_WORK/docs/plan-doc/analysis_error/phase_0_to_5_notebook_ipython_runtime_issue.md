# PHASE 0-5 NOTEBOOK IPYTHON RUNTIME ISSUE

## Issue metadata

```text
Issue ID: CW-PHASE-0005-N2-001
Parent plan: CW-REFACTOR-0005-001
Step: N2
Status: RESOLVED
Severity: BLOCKING_CURRENT_STEP
```

## Observed result

Notebook execution stopped before the kernel started because strict warning handling promoted the IPython unwritable-home warning to an exception.

```text
Notebook cells executed: 0
Notebook overwritten by nbconvert: false
Failure source: IPython runtime directory resolution
Requested user path: /Users/voanhnhat-ticoder-coder/.ipython
Writable project source required: false
Phase 0-5 artifacts modified: false
Phase 6 started: false
```

## Root cause

The managed filesystem permits project and temporary-directory writes but does not permit IPython to create runtime state under the user home directory. This is an execution-environment path issue, not a notebook, scientific-logic or kernel-contract defect.

## Required correction

Route IPython and Jupyter runtime, configuration and data directories to dedicated writable paths under `/private/tmp`, expose the verified virtual-environment kernelspec through `JUPYTER_PATH`, and rerun the unchanged notebook with strict warnings.

## Resolution evidence

```text
Dedicated temporary runtime directories created: true
User-home IPython warning repeated: false
Kernel resolution advanced to socket allocation: true
Notebook cells executed during recovery check: 0
Notebook source changed by recovery: false
```
