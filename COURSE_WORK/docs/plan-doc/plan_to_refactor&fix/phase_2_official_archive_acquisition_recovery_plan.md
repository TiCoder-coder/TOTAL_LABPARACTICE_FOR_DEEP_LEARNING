# PHASE 2 OFFICIAL ARCHIVE ACQUISITION RECOVERY PLAN

## Plan metadata

```text
Plan ID: CW-PHASE-0005-P2-FIX-001
Issue ID: CW-PHASE-0005-P2-001
Execution mode: SEQUENTIAL
Scope: Restore the AQ0 official-source gate
Status: COMPLETED
```

## Objective

Obtain the official UCI ID 374 archive without altering the protected canonical CSV, then resume the approved P2 integrity sequence.

## Recovery path A

Human approves the scoped `curl` action for the official UCI archive URL.

```text
Download destination:
COURSE_WORK/data/raw_data/source/appliances_energy_prediction.zip.part

Canonical rename allowed only after:
ZIP opens successfully
ZIP test passes
Expected CSV member is safe
Extracted CSV SHA-256 matches the protected baseline
```

## Recovery path B

Human manually downloads the official UCI archive and places it at:

```text
COURSE_WORK/data/raw_data/source/appliances_energy_prediction.zip.part
```

The same validation sequence remains mandatory. Manual placement does not bypass hash or archive-integrity checks.

## Validation sequence

```text
1. Verify Phase 0 and Phase 1 sign-offs.
2. Validate the ZIP without extracting all members.
3. Reject absolute paths and parent traversal.
4. Locate the expected energydata_complete.csv member.
5. Extract only that member to temporary storage.
6. Compare its SHA-256 with the protected CSV.
7. Stop on any mismatch without overwriting either file.
8. Rename the validated archive atomically.
9. Create provenance, checksums, metadata, README and acquisition artifacts.
10. Run P2 tests and upstream regression checks.
```

## Stop conditions

```text
Network permission remains unavailable and no manual archive exists.
ZIP integrity fails.
Expected member is missing or unsafe.
Extracted CSV differs from the protected CSV.
Official source identity cannot be verified.
```

No Phase 3 work is allowed until `phase_2_signoff.json` is `PASS`.
