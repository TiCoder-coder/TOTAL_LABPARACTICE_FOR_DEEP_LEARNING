# JUPYTER CURVE TRANSPORT COMPATIBILITY ISSUE

## Issue metadata

```text
Issue ID: CW-NOTEBOOK-IMPORT-002
Parent plan: CW-NOTEBOOK-IMPORT-FIX-001
Step: Clean notebook execution
Status: RESOLVED
Severity: BLOCKING_CURRENT_STEP
```

## Observed failure

The external Jupyter kernel was requested with required CurveZMQ encryption. Kernel provisioning created Curve keys, but client construction failed before the first notebook cell.

```text
jupyter_client: 8.9.1
ipykernel: 7.3.0
nbclient: 0.11.0
Failure: TraitError
Field: AsyncKernelClient.curve_publickey
Expected type: bytes
Received type: str
Notebook cells executed: 0
Canonical notebook overwritten: false
Temporary executed notebook created: false
```

## Root cause

The installed Jupyter transport stack has a type-compatibility defect on the required CurveZMQ provisioning path. This is independent of the coursework package import fix and occurs before notebook code starts.

## Constraints

```text
Do not upgrade or downgrade Jupyter dependencies in this fix.
Do not patch third-party site-packages.
Do not weaken notebook, Phase or artifact validation.
Do not expose the kernel on a non-local interface.
```

## Required correction

Use the standard compatible Jupyter transport bound to localhost, retain Jupyter HMAC message signing, place connection and runtime state in dedicated permission-restricted temporary directories, suppress only the transport advisory log, retain warnings-as-errors inside notebook code, and atomically write only after all cells pass.

## Resolution evidence

```text
Kernel transport: standard localhost Jupyter transport
Connection authentication: Jupyter HMAC signing retained
Runtime directory permissions: 0700
Dependency version changes: none
Third-party source patches: none
Notebook cells executed: 8
Notebook errors: 0
Notebook warning streams: 0
Atomic canonical write: completed after successful execution
Temporary executed notebook retained: false
```
