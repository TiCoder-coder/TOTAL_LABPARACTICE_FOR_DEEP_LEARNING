# Phase 34 — Dataset Manifest Recovery Plan

## Objective

Restore:

```text
data/raw_data/dataset_manifest.json
```

so that `load_upstream_context()` can complete without bypassing provenance checks.

## Recovery order

```text
1. exact Git/Git-LFS restore if available
2. deterministic reconstruction from canonical source evidence
3. documented blocker if exact recovery is impossible
```

## Audit requirements

Check:

```text
expected schema
expected fingerprint/checksum references
Git history
Git LFS
.gitignore
canonical generator/materializer
tests/contracts consuming dataset_manifest.json
downstream lineage fields depending on it
```

## Verification

Recovery succeeds only when:

```text
manifest reloads successfully
schema is valid
fingerprints/checksums are consistent
load_upstream_context() PASS
Phase 34 preflight remains PASS
H2 becomes READY
Test remains FORBIDDEN
```

Stop before H2 and let the human run training manually.
