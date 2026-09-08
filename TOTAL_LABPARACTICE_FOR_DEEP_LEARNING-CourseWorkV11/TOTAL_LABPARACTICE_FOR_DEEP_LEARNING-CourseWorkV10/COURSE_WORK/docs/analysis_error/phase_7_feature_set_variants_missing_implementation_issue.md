# Phase 7 Feature-Set Variants Missing Implementation Issue

## Issue ID

`CW-PHASE-7-MISSING-001`

## Status

`CONFIRMED`

## Current state

Phase 6 is signed off as `FEATURES-v1`, but Phase 7 has no source owner, no `FEATURESETS-v1` artifacts, no automated tests, and no notebook orchestration section. The architecture contract currently stops at Phase 6.

## Required state

Phase 7 must freeze six ordered variants, their components, counts, semantics and fingerprints without copying the dataset, selecting a winner, splitting, scaling, building windows or training models.

## Canonical owner

Create one new source file:

```text
src/course_work/data/feature_sets.py
```

This keeps Phase 6 feature creation in `data/features.py`, places Phase 7 feature-set semantics in a separate owner, and leaves `data/splitting.py` reserved for Phase 8.

## Specification discrepancy

The Phase 7 document requires `feature_set_manifest.json` in Sections 62 and 128, but omits it from the output-directory list in Section 117. The implementation must use the stricter superset and create the manifest in addition to every listed output.

## Constraints

- Build exactly `FS0_TF0`, `FS0_TF1`, `FS1_TF0`, `FS1_TF1`, `FS2_TF0`, `FS2_TF1`.
- Preserve explicit feature order.
- Use `FEATURES-v1`, Phase 6 registry, availability and leakage artifacts as source inputs.
- Do not add or transform feature values.
- Do not create six data copies.
- Keep metadata out of every model variant.
- Keep `Appliances` only in FS1 and FS2.
- Keep `rv1` and `rv2` only in FS2.
- Keep the five engineered time features only in TF1.
- Keep notebook Phase 7 orchestration-only.
- Do not add dependencies, code comments or icons.

## Resolution gate

The issue is resolved only when `FEATURESETS-v1` reloads with all six variants, all counts and fingerprints match, every audit passes, Phase 0-7 integration passes, the notebook executes cleanly, and the full test suite passes.
