# PHASE 1 KERNEL RESOLUTION ISSUE

## Issue metadata

```text
Issue ID: CW-PHASE-0005-P1-001
Parent plan: CW-REFACTOR-0005-001
Phase: 1
Status: RESOLVED
Severity: BLOCKING_CURRENT_PHASE
```

## Observed failure

```text
test_device_selection_follows_available_backend: PASS
test_device_smoke_test_passes: PASS
test_environment_inventory_has_required_fields: FAIL
test_kernel_matches_interpreter: FAIL
```

No Phase 1 artifact was written before the failure.

## Evidence

```text
Active interpreter:
TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/venv/bin/python

Resolved interpreter target:
/Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10

Existing kernel specification:
TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/venv/share/jupyter/kernels/python3/kernel.json

Kernel command:
python -m ipykernel_launcher
```

The notebook kernel name is `python3`. Its kernelspec exists inside the active virtual environment and resolves through that environment.

## Root cause

`resolve_kernel_contract()` resolved the interpreter symlink before deriving the virtual-environment kernelspec directory. It therefore searched below the base framework installation instead of below `sys.prefix` or the unresolved virtual-environment executable path.

This is a path-resolution defect in the new Phase 1 audit code. It is not evidence of an incorrect notebook kernel.

## Scope and integrity state

```text
Phase 0 sign-off: PASS
Phase 1 artifacts created: false
Raw data modified: false
Notebook modified: false
Phase 2 started: false
```

## Required correction

Kernel discovery must use the active environment prefix. Executable identity comparison must continue to use resolved paths so that virtual-environment symlinks and their base interpreter target compare correctly.

## Resolution evidence

```text
Kernelspec discovered under sys.prefix: true
Kernel executable matches active interpreter: true
Phase 1 unit tests: PASS
Phase 1 sign-off: PASS
Environment ID: ENV-v1
```
