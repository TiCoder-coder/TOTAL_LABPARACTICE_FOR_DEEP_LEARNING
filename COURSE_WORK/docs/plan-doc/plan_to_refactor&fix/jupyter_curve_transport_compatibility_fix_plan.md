# JUPYTER CURVE TRANSPORT COMPATIBILITY FIX PLAN

## Plan metadata

```text
Plan ID: CW-NOTEBOOK-IMPORT-FIX-002
Issue ID: CW-NOTEBOOK-IMPORT-002
Scope: One clean local notebook execution
Status: COMPLETED
```

## Implementation

```text
1. Verify the failed encrypted attempt left the notebook clean.
2. Create dedicated IPython and Jupyter runtime directories with mode 0700.
3. Start the verified python3 kernel on its default localhost transport.
4. Retain connection-file authentication and restrict the process to local execution.
5. Suppress the known transport advisory through kernel log level only.
6. Retain warnings-as-errors for notebook and project code.
7. Execute from the notebook directory without PYTHONPATH.
8. Write a sibling temporary notebook only after all cells complete.
9. Atomically replace the canonical notebook and run full validation.
```

## Acceptance criteria

```text
The package imports without path injection.
No kernel binds to a non-local interface.
No network data acquisition occurs.
All eight code cells execute in order.
No notebook error or warning output exists.
No third-party dependency changes.
All Phase 0-5 checksums and tests pass.
```

## Completion result

```text
The clean kernel execution completed on localhost without package-path injection.
All eight notebook code cells completed in order.
No error output, warning stream, dependency change or temporary notebook remains.
All parent-plan acceptance checks pass.
```
