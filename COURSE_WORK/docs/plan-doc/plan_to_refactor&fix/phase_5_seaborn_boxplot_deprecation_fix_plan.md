# PHASE 5 SEABORN BOXPLOT DEPRECATION FIX PLAN

## Plan metadata

```text
Plan ID: CW-PHASE-5-EDA-DIRECT-FIX-003
Issue ID: CW-PHASE-5-EDA-DIRECT-003
Status: COMPLETED
```

## Implementation

```text
1. Replace Seaborn boxplots with Matplotlib boxplots using current keyword arguments.
2. Preserve the same raw groups, titles, outlier visibility policy and before/after demonstration.
3. Update notebook coverage tests without weakening the direct EDA requirement.
4. Retain warnings-as-errors.
5. Run boundary tests and the complete notebook from a clean kernel.
```

## Acceptance criteria

```text
No pending-deprecation warning occurs.
Conditional target distributions remain visible.
Raw and demonstration boxplots remain visible.
No data or artifact is mutated.
All cells execute successfully.
```

## Completion result

Conditional and smoothing boxplots use the current Matplotlib API. The complete notebook executes with warnings-as-errors and produces no warning stream.
