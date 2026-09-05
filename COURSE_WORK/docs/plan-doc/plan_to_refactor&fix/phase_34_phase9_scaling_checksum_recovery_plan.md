# Phase 34 — Phase 9 Scaling Checksum Recovery Plan

## Objective

Restore exact signed Phase 9 scaling evidence required by:

```text
load_validated_target_scaler()
```

Current blocker:

```text
artifacts/scaling/scaling_audit.csv
```

## Recovery method

Audit all Phase 9 declared outputs in one pass.

For every Phase 9 output:

```text
check existence
calculate current SHA256
compare signed SHA256
classify PASS / MISSING / CHECKSUM_MISMATCH
```

Recovery order:

```text
1. exact Git/Git-LFS restore
2. exact deterministic reconstruction from canonical Train-only scaling evidence
3. documented blocker if exact recovery is impossible
```

Candidate reconstruction must be written to a temporary location first and restored only when the signed SHA256 matches exactly.

## Forbidden

```text
do not rewrite sign-off checksums
do not accept a different scaler revision
do not refit using Validation/Test
do not bypass scaling verification
do not run H2
do not access Test
```

## Success criteria

```text
all Phase 9 declared outputs verify
materialize_phase_9() PASS
load_validated_target_scaler() PASS
Phase 34 preflight PASS
H2 READY
Test FORBIDDEN
```

Stop before H2. The human runs training manually.
