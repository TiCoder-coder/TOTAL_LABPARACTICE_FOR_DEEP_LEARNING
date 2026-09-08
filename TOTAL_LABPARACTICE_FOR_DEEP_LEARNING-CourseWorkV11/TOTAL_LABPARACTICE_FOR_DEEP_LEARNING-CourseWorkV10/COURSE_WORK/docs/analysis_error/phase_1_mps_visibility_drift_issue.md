# PHASE 1 MPS VISIBILITY DRIFT ISSUE

## Issue metadata

```text
Issue ID: CW-PHASE-0005-N2-005
Parent plan: CW-REFACTOR-0005-001
Step: N2
Status: RESOLVED
Severity: BLOCKING_CURRENT_STEP
```

## Observed result

The repository-root executor removed working-directory drift, but running the kernel outside the managed sandbox exposed Apple MPS and no longer matched CPU-signed ENV-v1.

```text
Recorded mps_available: false
Current mps_available outside sandbox: true
Recorded selected_device: cpu
Current selected_device outside sandbox: mps
Other environment fields differing: none
Standard TCP kernel inside sandbox: blocked
Unix IPC kernel inside sandbox: blocked
```

## Root cause

The managed boundary changes hardware visibility. ENV-v1 was produced and validated inside the managed environment, while lifting the boundary solely to permit Jupyter sockets exposes an additional accelerator. Regenerating or weakening ENV-v1 would change a valid scientific environment contract to accommodate an executor transport limitation.

## Candidate directions

```text
Direction A: regenerate the entire signed chain under an MPS-visible ENV-v2.
Direction B: weaken Phase 1 hardware drift comparison.
Direction C: use ipykernel's in-process kernel on the signed interpreter inside the managed environment.
```

Direction A changes the approved experiment environment and cascades through Phase 2-5 sign-offs. Direction B hides real device drift. Direction C preserves the signed interpreter, CPU visibility, repository working directory, strict warnings and sequential IPython execution without network sockets.

## Required correction

Use Direction C. Execute each code cell once through `InProcessKernelManager`, capture canonical Jupyter messages into notebook outputs, reject any cell error or warning, and atomically write only after the complete Phase 0-5 sequence passes.

## Resolution evidence

```text
Runtime matches signed ENV-v1: true
Kernel interpreter: verified Phase 1 interpreter
Transport sockets used: none
Code cells executed: 8/8
Execution counts: 1, 2, 3, 4, 5, 6, 7, 8
Notebook cell errors: 0
Notebook warning streams: 0
Atomic notebook write: PASS
```
