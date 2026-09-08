# PHASE 0-5 NOTEBOOK KERNEL SOCKET ISSUE

## Issue metadata

```text
Issue ID: CW-PHASE-0005-N2-002
Parent plan: CW-REFACTOR-0005-001
Step: N2
Status: RESOLVED
Severity: BLOCKING_CURRENT_STEP
```

## Observed result

The verified Jupyter kernel could not allocate its localhost communication ports inside the managed sandbox.

```text
Notebook cells executed: 0
Notebook overwritten by nbconvert: false
Failure operation: socket.bind((ip, 0))
Failure type: PermissionError
Runtime-directory correction active: true
Phase 0-5 artifacts modified: false
Phase 6 started: false
```

## Root cause

Jupyter kernels require local sockets for shell, control, input, heartbeat and IOPub channels. The current filesystem sandbox also denies the required socket bind operation. This cannot be corrected in notebook or project source without replacing the validated Jupyter execution protocol.

## Required correction

Run the unchanged, already-scoped nbconvert command with the managed sandbox restriction lifted, retaining dedicated temporary runtime directories, the verified virtual-environment kernel, repository-root working directory and strict warnings.

## Resolution evidence

```text
Managed execution permission granted: true
Kernel process started: true
Local sockets allocated: true
Import cell reached: true
Notebook source changed by correction: false
```
