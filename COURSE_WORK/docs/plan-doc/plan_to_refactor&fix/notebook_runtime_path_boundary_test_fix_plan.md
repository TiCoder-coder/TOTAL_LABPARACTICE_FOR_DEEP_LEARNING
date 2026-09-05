# NOTEBOOK RUNTIME PATH BOUNDARY TEST FIX PLAN

## Plan metadata

```text
Plan ID: CW-PHASE-0005-N2-FIX-006
Issue ID: CW-PHASE-0005-N2-006
Scope: One notebook portability assertion
Status: COMPLETED
```

## Implementation

```text
1. Preserve the executed notebook and Phase 1 outputs.
2. Build the portability assertion input from cell source only.
3. Reject /Users/ in Markdown and code source.
4. Do not exclude or normalize any runtime evidence output.
5. Rerun the notebook boundary test.
6. Rerun the complete Phase 0-5 test suite.
7. Revalidate notebook execution state and checksums.
```

## Acceptance criteria

```text
Machine-specific paths remain forbidden in notebook source.
Required Phase 1 runtime paths remain visible in generated output.
All five boundary tests pass.
All full-suite tests pass.
No notebook or scientific source change is made.
```

## Completion result

```text
The assertion now rejects machine-specific paths in notebook source only.
Required environment-report paths remain visible in executed output.
All five notebook boundary tests pass.
All 42 Phase 0-5 tests pass.
```
