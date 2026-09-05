# PHASE 5 DIRECT EDA HEADLESS DISPLAY ISSUE

## Issue metadata

```text
Issue ID: CW-PHASE-5-EDA-DIRECT-002
Parent plan: CW-PHASE-5-EDA-DIRECT-001
Status: RESOLVED
Severity: BLOCKING_NOTEBOOK_EXECUTION
```

## Failure

The clean notebook kernel reached the direct missingness figure and failed when `plt.show()` attempted to use the non-interactive `FigureCanvasAgg` backend while warnings were configured as errors.

```text
Failure type: UserWarning promoted to exception
Message: FigureCanvasAgg is non-interactive, and thus cannot be shown
Failed cell: phase-5-missingness-heatmap
Canonical notebook overwritten: false
Scientific calculation failure: false
```

## Root cause

`plt.show()` delegates to the active GUI backend. Headless verification intentionally uses a non-interactive backend, so the call emits a warning. The figure object itself is valid and can be rendered through the notebook display protocol without a GUI manager.

## Required correction

Display each direct Matplotlib figure through `IPython.display.display(figure)` and explicitly close it after display. Retain warnings-as-errors and rerun the complete notebook from a clean kernel.

## Resolution evidence

```text
Direct plt.show calls: 0
Notebook image outputs: 16
Notebook warning streams: 0
Notebook errors: 0
```
