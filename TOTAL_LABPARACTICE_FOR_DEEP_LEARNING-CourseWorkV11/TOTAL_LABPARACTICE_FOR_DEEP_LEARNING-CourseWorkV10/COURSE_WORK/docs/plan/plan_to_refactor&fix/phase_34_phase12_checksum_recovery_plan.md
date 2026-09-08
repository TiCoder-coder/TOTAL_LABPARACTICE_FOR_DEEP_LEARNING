# Phase 34 — Phase 12 Checksum Recovery Plan

## Objective

Restore the exact signed Phase 12 artifact required by `load_upstream_context()`:

```text
artifacts/metrics/metric_implementation_audit.csv
```

## Recovery order

```text
1. exact Git restore if a matching signed artifact exists;
2. deterministic reconstruction if the canonical source can reproduce the signed SHA256;
3. documented corrective recovery if neither is possible.
```

## Read-only audit first

Determine:

- signed expected SHA256;
- current SHA256;
- Git history and Git object availability;
- whether `.gitignore` affected retention;
- canonical generator/materializer for the CSV;
- whether mismatch is content drift vs serialization/newline drift.

## Forbidden

```text
do not rewrite Phase 12 sign-off checksum
do not fabricate CSV content
do not bypass verification
do not start H2 while mismatch remains
do not access Test
```

## Success criteria

```text
current SHA256 == signed SHA256
materialize_phase_12() PASS
load_upstream_context() PASS
Phase 34 preflight PASS
Test remains FORBIDDEN
```

After recovery, stop before H2 and let the human rerun it manually.
