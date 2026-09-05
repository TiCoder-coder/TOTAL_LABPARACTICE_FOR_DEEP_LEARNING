# COURSEWORK NOTEBOOK SRC-LAYOUT IMPORT ISSUE

## Issue metadata

```text
Issue ID: CW-NOTEBOOK-IMPORT-001
Scope: CourseWork.ipynb, project packaging, Phase 1 environment validation
Status: RESOLVED
Severity: BLOCKING_NOTEBOOK
```

## Observed failure

The first notebook code cell fails at the first `course_work` import.

```text
Exception: ModuleNotFoundError
Missing module: course_work
Kernel name: python3
Kernel environment: repository venv
Canonical source root: COURSE_WORK/src
Package source exists: true
Package installed in kernel environment: false
Project pyproject.toml exists: false
```

The notebook now contains one failed execution count and stale outputs from an earlier successful execution in later cells. Its stored execution state is internally inconsistent.

## Layered root-cause analysis

### Layer 1 - Notebook

The import statements use the correct canonical package name. Adding `sys.path` mutation or path-discovery logic to the notebook would violate the approved orchestration-only boundary.

### Layer 2 - Python import system

Python searches the kernel working directory and installed site-packages. It does not automatically add a sibling `COURSE_WORK/src` directory to `sys.path`.

### Layer 3 - Project packaging

The repository uses a canonical src layout but has no active `pyproject.toml`. Therefore the source tree is not installable as a project package and the kernel has no editable-package registration.

### Layer 4 - Kernel environment

The selected kernelspec correctly resolves to the repository virtual environment, but `importlib.util.find_spec("course_work")` returns `None`. Kernel identity and package availability are separate contracts.

### Layer 5 - Previous execution method

The earlier automated execution injected `COURSE_WORK/src` through a temporary `PYTHONPATH`. That made one executor pass but did not configure interactive VS Code/Jupyter sessions. The successful stored outputs therefore did not prove standalone notebook importability.

### Layer 6 - Phase 1 follow-on drift

After import is corrected, current `materialize_phase_1()` compares every inventory field against signed ENV-v1. Working directory, accelerator visibility and selected device are runtime-context fields and can legitimately differ between managed execution and VS Code while Python, dependencies and kernel identity remain stable. Strict full-dictionary equality can produce a second false failure.

## Constraints

```text
Preserve Phase 0 through Phase 5 order.
Preserve source-owned processing.
Keep the notebook orchestration-only.
Do not add sys.path or PYTHONPATH bootstrap code to the notebook.
Do not move or duplicate course_work outside COURSE_WORK/src.
Do not alter scientific contracts, dataset bytes or signed Phase 2-5 outputs.
Do not add third-party dependencies.
Do not start Phase 6.
```

## Required resolution

Create standard project packaging for the existing src layout, install it editable into the verified virtual environment, distinguish stable environment identity from runtime context during ENV-v1 verification, show the current runtime inventory in the notebook, clear inconsistent state, execute top to bottom, and revalidate all Phase 0-5 artifacts and tests.

## Resolution evidence

```text
Editable package: course-work 0.1.0
Resolved module: COURSE_WORK/src/course_work/__init__.py
Notebook PYTHONPATH injection: absent
Notebook sys.path mutation: absent
Executed code-cell counts: 1, 2, 3, 4, 5, 6, 7, 8
Notebook errors: 0
Notebook warning streams: 0
Automated tests: 46 passed
External dependency freeze: unchanged
Raw dataset SHA-256: 2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d
Phase 0-5 sign-off checksums: unchanged
Phase 6 artifacts: absent
```
