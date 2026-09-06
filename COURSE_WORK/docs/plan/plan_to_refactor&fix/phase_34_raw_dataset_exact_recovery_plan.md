# Phase 34 — Raw Dataset Exact Recovery Plan

## Objective

Restore the exact protected raw dataset:

```text
data/raw_data/energydata_complete.csv
```

Expected SHA256:

```text
2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d
```

## Recovery Order

```text
1. exact Git restore if tracked
2. exact Git LFS restore if tracked
3. exact local/archive/backup restore
4. exact original-source copy only if SHA256 matches
5. otherwise STOP
```

## Read-Only Audit

Check:

```text
Git history
Git LFS
.gitignore
local duplicates in repository/project copies
archive/history/backup locations
dataset manifest/source metadata
any canonical acquisition script or documented source
```

## Forbidden

```text
do not reconstruct raw data from processed/interim data
do not alter the manifest checksum
do not accept a different dataset revision
do not run H2
do not access Test
```

## Verification

Recovery succeeds only when:

```text
sha256(data/raw_data/energydata_complete.csv)
==
2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d
```

Then verify:

```text
load_upstream_context() PASS
Phase 34 preflight PASS
H2 READY
Test FORBIDDEN
```

Stop before H2; the human runs training manually.
