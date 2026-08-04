# Phase 6 — Verification

## Active authority

- Registry status: **active**.
- Default dataset, manifest, E2 checkpoint and canonical output paths: **NEW LAYOUT**.
- Explicit legacy fallback: **PASS**, read-only and never default.
- Dataset fingerprint: `26dc4625f96c7cb86bc6a4df0fbed34bc8b22fb6472ef0d008df341f99f71fd6`.
- Split fingerprint: `52aaf97499ede5dd4689c2bb36dc0679fe04b8ec8139aae7e8fdcde88042043d`.
- Model-use counts: Train 2,016; Validation 438; Test 440.
- Generated excluded: 306; quarantined: 2.

## Validation-only reload

The active dataset, manifest and copied E2 checkpoint were loaded. Only a
Validation DataLoader was constructed.

| Metric | Result |
|---|---:|
| Validation samples | 438 |
| Accuracy | 78.31050228310502% |
| Loss | 1.0217703706053294 |
| Macro F1 | 0.7820960879325867 |
| Test DataLoader constructed | false |
| Final Test evaluated | false |

## Notebook and imports

- `notebooks/04_canonical_report.ipynb` Run All: **PASS**, 19/19 code cells executed, 0 error outputs.
- Notebook training / optimizer / Test-loader operations: **none**.
- Static HTML regenerated from the executed notebook: **PASS**.
- New `practice_2_2.*` import: **PASS** and recommended/default.
- Old `processing_own_phase.*` regression import: **PASS**.

## Automated tests

- New Practice 2.2 suite: **50 passed in 3.56s**.
- Full old Practice 2 suite: **87 passed, 3 deprecation warnings in 9.16s**.

## Lineage safety

All recorded immutable hashes are identical before and after the authority
switch and notebook Run All. The Final Test completion guard remains locked:
one historical evaluation, repeat disabled, maintenance override false.

No training, Test DataLoader construction, Final Test inference, archive,
delete or canonical immutable-artifact rewrite occurred in Phase 6.
