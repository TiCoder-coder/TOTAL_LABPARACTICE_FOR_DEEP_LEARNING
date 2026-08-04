# Phase 5 — Canonical Resource Copy Verification

## Copy integrity

All eight source/destination directory pairs have identical relative file sets, file counts, total bytes and per-file SHA-256 values. Deterministic directory digests match for every pair.

| Group | Files | Bytes | Result |
|---|---:|---:|---|
| Dataset | 3,202 | 48,502,262 | BYTE-IDENTICAL |
| Canonical split | 3 | 1,067,486 | BYTE-IDENTICAL |
| Canonical E1/E2 checkpoints | 4 | 313,702,500 | BYTE-IDENTICAL |
| Canonical outputs | 16 | 335,691 | BYTE-IDENTICAL |
| E3 checkpoints | 2 | 165,229,782 | BYTE-IDENTICAL |
| E3 outputs | 4 | 7,360 | BYTE-IDENTICAL |
| E4 checkpoints | 2 | 224,014,206 | BYTE-IDENTICAL |
| E4 outputs | 4 | 10,219 | BYTE-IDENTICAL |
| Quarantine CSV | 1 | 914 | BYTE-IDENTICAL |

Quarantined images remain members of the copied dataset tree and are excluded by the unchanged manifest. No new quarantine image was generated.

## Draft registry

- File: `configs/canonical_registry_phase5_draft.json`
- Authority: `verified_copy_not_active`
- Active registry still resolves `data_clean_balanced` and the original run/output trees.
- Draft registry resolves only the verified copies.

## Copied-resource validation

- Dataset fingerprint: PASS.
- Split fingerprint: PASS.
- Manifest model-use counts: Train 2016, Validation 438, Test 440.
- Generated excluded: 306.
- Quarantined: 2.
- E2 checkpoint SHA-256: PASS.
- Final selection, summary, lineage and guard hashes: PASS.
- E3/E4 checkpoint and run-ID verification: PASS.

## Copied checkpoint reload Validation

| Metric | Result |
|---|---:|
| Validation samples | 438 |
| Accuracy | 78.31050228310502% |
| Accuracy delta | 0.0 |
| Loss | 1.0217703706053294 |
| Loss delta | 0.0000000103423585 |
| Macro F1 | 0.7820960879325867 |
| Test DataLoader constructed | false |
| Final Test evaluated | false |

The tiny loss delta is floating-point evaluation noise; Accuracy and Macro F1 match exactly.

## Automated tests

- New layout, registry and copy-integrity suite: **47 passed in 12.51s**.
- Full old Practice 2 regression suite: **87 passed, 3 deprecation warnings in 9.06s**.
- Old/new import smoke tests remain covered and pass.

## Safety outcome

- Source resource digests remain equal to the pre-copy inventory.
- No source file was deleted, renamed or modified.
- No persisted canonical JSON was rewritten.
- No canonical authority was switched.
- No training or Final Test inference occurred.
