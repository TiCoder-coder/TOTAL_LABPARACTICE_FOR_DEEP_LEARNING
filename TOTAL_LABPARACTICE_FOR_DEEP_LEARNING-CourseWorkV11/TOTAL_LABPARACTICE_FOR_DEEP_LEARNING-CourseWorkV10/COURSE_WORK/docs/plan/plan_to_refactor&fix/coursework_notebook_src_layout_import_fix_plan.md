# COURSEWORK NOTEBOOK SRC-LAYOUT IMPORT FIX PLAN

## Plan metadata

```text
Plan ID: CW-NOTEBOOK-IMPORT-FIX-001
Issue ID: CW-NOTEBOOK-IMPORT-001
Execution mode: STRICTLY_SEQUENTIAL
Status: COMPLETED
```

## Objective

Make `CourseWork.ipynb` independently executable from VS Code/Jupyter with the verified project kernel while preserving the existing Phase 0-5 flow and architecture.

## Impacted files

```text
COURSE_WORK/pyproject.toml
COURSE_WORK/README.md
COURSE_WORK/docs/RULE_BASE/architecture_rule.md
COURSE_WORK/src/course_work/utils/environment.py
COURSE_WORK/notebook_course_work/CourseWork.ipynb
COURSE_WORK/tests/unit/test_environment.py
COURSE_WORK/tests/integration/test_notebook_boundary.py
```

Environment action:

```text
Install the local COURSE_WORK project editable into the existing repository venv.
Use no dependency download and no dependency upgrade.
```

## Implementation

### Step 1 - Preserve and verify baseline

```text
Record the raw CSV SHA-256.
Record all Phase 0-5 sign-off checksums.
Record notebook source and failed execution state.
Confirm Phase 6 remains unstarted.
```

### Step 2 - Activate canonical src-layout packaging

```text
Create pyproject.toml using the existing COURSE_WORK/src package root.
Use setuptools already installed in the venv.
Declare no runtime dependency changes in pyproject.toml.
Configure package discovery for course_work and its existing packages only.
Document the editable-install command in README.md.
Register pyproject.toml ownership in architecture_rule.md without changing the Phase flow.
```

### Step 3 - Install locally

```text
Run pip install --editable COURSE_WORK --no-deps --no-build-isolation.
Verify importlib can resolve course_work from the notebook directory with PYTHONPATH absent.
Verify the resolved module path belongs to COURSE_WORK/src/course_work.
```

### Step 4 - Correct Phase 1 identity validation

```text
Define a stable environment identity from Python, core package versions, platform, dtype, seed policy and kernel identity.
Exclude working directory, project location, processor count, accelerator availability and selected device from stable identity equality.
Continue collecting and displaying every runtime-context field.
Fail on stable dependency or kernel drift.
Allow runtime-context differences without rewriting ENV-v1.
Exclude the local editable project from the external dependency freeze.
```

### Step 5 - Correct notebook runtime reporting

```text
Import the public environment_inventory API.
Display the current runtime inventory instead of reloading the historical signed report as the current runtime.
Keep all Phase calls and their order unchanged.
Clear every stale output and execution count before validation.
```

### Step 6 - Add regression coverage

```text
Test that runtime-context differences do not change stable identity.
Test that Python/package/kernel identity drift remains blocking.
Test that the editable local project is excluded from dependency freeze.
Test package importability from the notebook directory without PYTHONPATH.
Keep all existing notebook boundary tests.
```

### Step 7 - Verify sequentially

```text
Run packaging and import checks.
Run Phase 1 unit tests.
Run notebook boundary tests.
Run the complete Phase 0-5 suite with warnings as errors.
Clear and execute CourseWork.ipynb top to bottom in a clean kernel context.
Verify monotonic execution counts and zero error or warning outputs.
Verify all signed output checksums.
Verify the raw CSV hash is unchanged.
Verify Phase 6 remains unstarted.
```

## Alternatives rejected

```text
Notebook sys.path mutation: violates orchestration-only ownership.
Notebook PYTHONPATH bootstrap: hidden environment state and machine fragility.
Hard-coded kernelspec environment path: machine-specific and non-portable.
Move package beside notebook: duplicates and breaks canonical architecture.
Ignore all ENV-v1 drift: would hide real dependency or kernel changes.
Regenerate Phase 2-5 artifacts: unnecessary scientific-scope expansion.
```

## Risks and controls

```text
Editable install changes pip freeze output: use pip freeze --exclude-editable and verify it remains byte-identical to the signed external dependency freeze.
Runtime drift tolerance could be too broad: compare an explicit stable-field allowlist and test blocking changes.
Notebook may retain mixed outputs after a failure: clear all state before clean execution and write only after successful completion.
Packaging could include reserved Phase 6 namespaces: setuptools finds only directories that are existing Python packages; no Phase 6 implementation is added.
```

## Acceptance criteria

```text
course_work imports from the notebook directory without PYTHONPATH.
The import resolves to COURSE_WORK/src/course_work.
The notebook contains no path bootstrap or processing logic.
Phase 0-5 order and public calls are unchanged.
Stable ENV-v1 drift remains blocking.
Runtime-context drift is reported but not falsely blocking.
All notebook cells execute once in order.
No error or warning output exists.
All automated tests pass.
All signed checksums match.
Raw data remains unchanged.
Phase 6 remains unstarted.
```

## Completion result

```text
All implementation steps completed sequentially.
Canonical src-layout packaging is active through an editable local install.
Phase 1 validates stable environment identity while reporting current runtime context.
CourseWork.ipynb executes all eight code cells in order without errors or warning streams.
The full automated suite passes with 46 tests.
Scientific inputs, signed Phase 0-5 outputs and the approved phase flow remain unchanged.
```
