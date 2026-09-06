# PHASE 1 MPS VISIBILITY DRIFT FIX PLAN

## Plan metadata

```text
Plan ID: CW-PHASE-0005-N2-FIX-005
Issue ID: CW-PHASE-0005-N2-005
Scope: Notebook kernel transport only
Status: COMPLETED
```

## Implementation

```text
1. Keep the canonical notebook clean until full execution succeeds.
2. Start ipykernel's in-process kernel on the active Phase 1 interpreter.
3. Execute code cells in notebook order with history enabled.
4. Capture stream, display, execute-result and error messages using the Jupyter message protocol.
5. Reject non-OK shell replies, error outputs and unresolved warnings.
6. Set execution counts from actual kernel execution-input messages.
7. Validate notebook JSON and atomically replace the canonical notebook only after success.
8. Revalidate Phase 0-5 source tests, signed outputs and raw checksum.
```

## Acceptance criteria

```text
The runtime matches signed ENV-v1 exactly.
All code cells execute once in increasing order.
All displayed figures come from signed Phase 5 artifacts.
No TCP or IPC transport is used.
No error or warning output exists.
No signed environment contract is changed or weakened.
Raw data remains unchanged.
Phase 6 remains unstarted.
```

## Completion result

```text
All eight notebook code cells executed through ipykernel in process.
ENV-v1 matched without regeneration or weakened validation.
Execution counts are strictly increasing from 1 through 8.
No error or warning output was captured.
The notebook was atomically written only after complete success.
```
