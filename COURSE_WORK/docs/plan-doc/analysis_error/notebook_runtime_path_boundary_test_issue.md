# NOTEBOOK RUNTIME PATH BOUNDARY TEST ISSUE

## Issue metadata

```text
Issue ID: CW-PHASE-0005-N2-006
Parent plan: CW-REFACTOR-0005-001
Step: N2 final validation
Status: RESOLVED
Severity: BLOCKING_FINAL_VALIDATION
```

## Observed result

The executed notebook passed all cells, but one boundary assertion rejected `/Users/` found in the displayed Phase 1 environment report.

```text
Notebook code hard-coded machine path: false
Notebook Markdown hard-coded machine path: false
Phase 1 output records runtime path: true
Failing assertion scope: entire serialized notebook including outputs
Boundary tests passed before execution: 5/5
Boundary tests passed after execution: 4/5
```

## Root cause

The portability rule forbids machine-specific paths in notebook source. Phase 1 separately requires machine-readable interpreter, kernel, project-root and working-directory evidence. The test incorrectly applied the source rule to generated evidence after notebook execution.

## Required correction

Limit the machine-specific path assertion to Markdown and code source while preserving all Phase 1 runtime evidence in outputs. Keep every other orchestration and execution-state assertion unchanged.

## Resolution evidence

```text
Portability assertion scope: Markdown and code source
Phase 1 runtime evidence preserved: true
Notebook boundary tests: 5/5 PASS
Full Phase 0-5 tests: 42/42 PASS
Notebook source changed: false
Scientific source changed: false
```
