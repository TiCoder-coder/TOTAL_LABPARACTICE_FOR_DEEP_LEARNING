# PHASE 5 SEABORN BOXPLOT DEPRECATION ISSUE

## Issue metadata

```text
Issue ID: CW-PHASE-5-EDA-DIRECT-003
Parent plan: CW-PHASE-5-EDA-DIRECT-001
Status: RESOLVED
Severity: BLOCKING_NOTEBOOK_EXECUTION
```

## Failure

Seaborn boxplot invoked Matplotlib's pending-deprecated `vert` argument. The project validation policy promotes warnings to errors, so execution stopped in the conditional target-distribution cell.

```text
Failure type: PendingDeprecationWarning
Failed cell: phase-5-conditional-boxplots
Canonical notebook overwritten: false
Completed EDA sections before failure: 5.1 through 5.9
```

## Required correction

Use Matplotlib's current boxplot API with explicit `orientation` and `tick_labels` for the conditional and smoothing demonstration figures. Do not suppress or ignore the dependency warning.

## Resolution evidence

```text
Deprecated Seaborn boxplot calls: 0
Warnings suppressed: 0
Notebook warning streams: 0
Conditional and smoothing figures rendered: true
```
