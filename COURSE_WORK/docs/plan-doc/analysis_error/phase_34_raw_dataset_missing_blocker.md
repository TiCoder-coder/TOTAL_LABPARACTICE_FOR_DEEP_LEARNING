# Phase 34 — Raw Dataset Missing Blocker

## Context

Phase 34 recovery has progressed to:

```text
Phase 12 = PASS
dataset_manifest.json = EXACT_RECONSTRUCTION / PASS
Phase 34 preflight = PASS
load_upstream_context() = BLOCKED
H2 = NOT_AUTHORIZED
Test = FORBIDDEN
```

The remaining blocker is the protected raw dataset:

```text
data/raw_data/energydata_complete.csv
```

Expected SHA256:

```text
2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d
```

## Interpretation

This is a raw-source integrity blocker, not an H2 training failure.

The dataset must not be synthesized, regenerated from processed data, or replaced by a different copy unless its SHA256 matches the signed expected value exactly.

## Current Decision

```text
Phase 34 = BLOCKED
H2 = NOT AUTHORIZED
Test = FORBIDDEN
```

Proceed only with an exact recovery audit for the raw dataset.
