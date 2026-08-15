# PHASE 0-5 NOTEBOOK KERNEL IMPORT PATH ISSUE

## Issue metadata

```text
Issue ID: CW-PHASE-0005-N2-003
Parent plan: CW-REFACTOR-0005-001
Step: N2
Status: RESOLVED
Severity: BLOCKING_CURRENT_STEP
```

## Observed result

The kernel reached the first code cell but could not import the source package.

```text
Executed cell: public-api-imports
Failure: ModuleNotFoundError: No module named 'course_work'
PYTHONPATH value supplied: COURSE_WORK/src
PYTHONPATH form: relative
Kernel working directory: notebook resource directory
Notebook overwritten by nbconvert: false
Phase artifacts modified: false
```

## Root cause

The shell resolved the relative `PYTHONPATH` only as an environment string. The child kernel interpreted that string relative to its own working directory, where `COURSE_WORK/src` does not identify the source root.

## Retrospective after three N2 attempts

### 1. Verified facts

```text
The notebook is valid and boundary-compliant.
The Phase 1 interpreter and kernel match.
The temporary runtime-directory correction works.
The managed socket permission works.
The source package imports in the verified interpreter when the source root is correctly exposed.
```

### 2. Invalidated assumptions

```text
An inherited relative PYTHONPATH is not stable across the parent process and kernel working directories.
Strict execution prerequisites must include child-process path resolution, not only parent-shell import checks.
```

### 3. Failure sequence

```text
Attempt 1: blocked by unwritable user-home IPython directory.
Attempt 2: blocked by sandbox localhost socket restriction.
Attempt 3: kernel started but relative source import path was invalid.
```

### 4. Candidate directions

```text
Direction A: pass the existing source root as an absolute PYTHONPATH to the kernel environment.
Direction B: add sys.path mutation or path-discovery code to the notebook.
Direction C: install the project as an editable package before execution.
```

Direction A preserves the approved notebook boundary and changes no project file. Direction B violates the orchestration-only and path-discovery rules. Direction C requires packaging configuration and environment mutation outside the approved Phase 0-5 scope.

### 5. Selected recovery and validation

Use Direction A, retain the corrected runtime and socket settings, execute the unchanged notebook once, then verify execution state, sign-offs, checksums, tests, raw immutability and the Phase 6 boundary.

## Resolution evidence

```text
Absolute source root supplied: true
Public API import cell executed: PASS
Phase 0 orchestration cell executed: PASS
ModuleNotFoundError repeated: false
Notebook source changed by correction: false
```
