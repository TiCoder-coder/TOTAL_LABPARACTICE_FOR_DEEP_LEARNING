# PHASE 5 HEADLESS FIGURE EMBEDDING FIX PLAN

## Plan metadata

```text
Plan ID: CW-PHASE-5-EDA-DIRECT-FIX-004
Issue ID: CW-PHASE-5-EDA-DIRECT-004
Status: COMPLETED
```

## Implementation

```text
1. Add one Phase 5 notebook helper that serializes a figure into an in-memory PNG.
2. Replace all direct raw-figure displays with the helper.
3. Keep canonical Image artifact displays unchanged.
4. Add a notebook boundary assertion for the in-memory display path.
5. Execute the notebook from a clean kernel.
6. Require at least 24 embedded PNG outputs: 16 canonical and 8 direct.
```

## Acceptance criteria

```text
All eight direct figures have image/png output.
All 16 canonical figures remain embedded.
No direct figure is written to disk.
No error or warning output exists.
```

## Completion result

All eight direct figures are serialized to in-memory PNG data and embedded in the notebook. Together with the 16 canonical figures, the executed notebook contains 24 PNG outputs and no warning or error stream.
