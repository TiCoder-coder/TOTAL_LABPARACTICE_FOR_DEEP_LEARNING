# PHASE 5 HEADLESS FIGURE EMBEDDING ISSUE

## Issue metadata

```text
Issue ID: CW-PHASE-5-EDA-DIRECT-004
Parent plan: CW-PHASE-5-EDA-DIRECT-001
Status: RESOLVED
Severity: PRESENTATION_NONCONFORMANCE
```

## Finding

The complete notebook executed successfully, but headless validation showed that direct Matplotlib figure objects produced only `text/plain` output. The 16 canonical PNG artifacts were embedded correctly, while the eight new direct figures were not stored as PNG notebook output.

## Root cause

The headless kernel does not register an inline PNG formatter for raw Matplotlib figure objects. `display(figure)` therefore emits a textual representation without raising an exception.

## Required correction

Serialize each direct figure to an in-memory PNG buffer, embed the PNG through `IPython.display.Image`, close the figure and rerun the full notebook. No filesystem artifact may be created by this correction.

## Resolution evidence

```text
Canonical PNG outputs: 16
Direct PNG outputs: 8
Total embedded PNG outputs: 24
Filesystem figure writes from direct cells: 0
Notebook errors: 0
Notebook warning streams: 0
```
