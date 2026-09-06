# Phase 22 Recovery Inspector False Block Issue

## 1. Issue identity

```text
Issue ID: CW-PHASE-22-RECOVERY-INSPECTOR-FALSE-BLOCK-001
Detected during: Phase 13 to Phase 21 recovery Step 9
Status: OPEN
Severity: BLOCKING
Date: 2026-08-22
```

## 2. Observed state

Fresh Phase 20 and Phase 21 sign-offs pass their canonical owner verifiers. Their registry records are completed, all required artifacts exist, all declared checksums match and the Test firewall passes.

The Phase 22 recovery inspector still classified both baselines as invalid because each sign-off includes its own path in `output_paths` without including an impossible recursive self-checksum in `output_checksums`.

The environment inspector also reported MPS unavailable after two completed MPS training runs. This must block later training phases but must not block Phase 22, which derives diagnostics from existing histories without training.

## 3. Root cause

`_validate_outputs` treated every unsigned output path as invalid even when the canonical phase owner intentionally leaves dynamic or self-referential outputs unsigned.

`run_pending` applied the accelerator training gate before the non-training Phase 22 recovery stage.

## 4. Required resolution

```text
Require every output path to exist
Verify every declared checksum
Allow an existing output path without a declared checksum
Recover Phase 22 before evaluating the accelerator gate for Phase 23 onward
Keep accelerator enforcement unchanged for all training phases
Keep Test access forbidden
```

## 5. Current gate

```text
Phase 20 canonical: true
Phase 21 canonical: true
Phase 22 recovery started: false
Phase 23 training authorized: false
```
