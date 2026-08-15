# PHASE 5 MATPLOTLIB BOXPLOT ORIENTATION FIX PLAN

## Plan metadata

```text
Plan ID: CW-PHASE-0005-P5-FIX-001
Issue ID: CW-PHASE-0005-P5-001
Scope: Three Phase 5 boxplot calls
Status: COMPLETED
```

## Implementation

```text
1. Replace legacy horizontal-orientation arguments with the current API.
2. Do not alter computed EDA values, labels, filenames or scientific scope.
3. Recompute P5 analysis tests.
4. Re-run materialization with warnings promoted to errors.
5. Byte-verify existing unsigned partial tables and figures.
6. Create manifest and sign-off only after all 16 figures exist.
7. Revalidate Phase 0 through Phase 4 and the raw checksum.
```

## Acceptance criteria

```text
No Matplotlib warning remains.
All 16 canonical figures exist.
All mandatory tables are non-empty.
Unsigned partial artifacts match regenerated content.
EDA-v1 sign-off is PASS.
Raw data remains unchanged.
```

## Completion result

```text
No Matplotlib warning remains.
All 16 canonical figures exist and passed structural verification.
All 16 canonical tables exist and are non-empty.
Phase 5 sign-off is PASS.
Phase 0-4 regression verification passed.
Raw data checksum is unchanged.
```
