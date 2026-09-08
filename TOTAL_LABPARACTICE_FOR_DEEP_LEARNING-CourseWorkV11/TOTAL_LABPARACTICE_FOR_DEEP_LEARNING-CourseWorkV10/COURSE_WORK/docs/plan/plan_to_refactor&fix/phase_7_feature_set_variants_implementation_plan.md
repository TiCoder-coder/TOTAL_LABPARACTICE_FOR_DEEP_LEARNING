# Phase 7 Feature-Set Variants Implementation Plan

## Plan ID

`CW-PHASE-7-IMPLEMENT-001`

## Approval status

`APPROVED_BY_CURRENT_HUMAN_REQUEST`

## Objective

Implement a single canonical Phase 7 module that validates `FEATURES-v1`, freezes the six registered feature-set variants, creates deterministic fingerprints and produces reusable `FEATURESETS-v1` artifacts for downstream phases.

## Inputs

- `artifacts/features/phase_6_signoff.json`
- `artifacts/features/feature_engineering_manifest.json`
- `artifacts/features/feature_registry.csv`
- `artifacts/features/feature_lineage.csv`
- `artifacts/features/feature_availability.csv`
- `artifacts/features/feature_leakage_audit.csv`
- `data/interim/uci_appliances_energy_prediction/energydata_feature_engineered_v1.csv`
- `configs/base/coursework_contract.json`

## Outputs

- `artifacts/feature_sets/feature_set_manifest.json`
- `artifacts/feature_sets/feature_set_registry.json`
- `artifacts/feature_sets/feature_set_registry.csv`
- `artifacts/feature_sets/feature_set_variants.csv`
- `artifacts/feature_sets/feature_components.json`
- `artifacts/feature_sets/feature_set_lineage.csv`
- `artifacts/feature_sets/feature_set_leakage_audit.csv`
- `artifacts/feature_sets/feature_order_checks.csv`
- `artifacts/feature_sets/feature_set_discrepancies.json`
- `artifacts/feature_sets/README_FEATURE_SETS.md`
- `artifacts/feature_sets/phase_7_signoff.json`

## Impacted files

- `src/course_work/data/feature_sets.py`
- `tests/unit/test_feature_sets.py`
- `tests/integration/test_phase_0_to_5_chain.py`
- `tests/integration/test_notebook_boundary.py`
- `notebook_course_work/CourseWork.ipynb`
- `docs/RULE_BASE/architecture_rule.md`

## Implementation sequence

### Step 1: Implement immutable components and variant construction

Define ordered tuple components for exogenous features, historical target, random controls, time features and metadata. Build all six variants only by ordered tuple concatenation and expose `get_feature_list` as a defensive-copy public API.

Verification:

- Exact expected counts are 25, 30, 26, 31, 28 and 33.
- Pairwise component differences match the Phase 7 contract.
- Building twice returns identical variants and fingerprints.
- Mutating a returned list cannot mutate the canonical registry.

### Step 2: Implement input validation and semantic audits

Verify Phase 6 sign-off and every signed checksum, reload `FEATURES-v1`, validate Phase 6 registries, and audit each variant for existence, duplicates, count, order, metadata exclusion, historical-target isolation, random-control isolation, time-feature isolation, model eligibility, prediction-time availability, numeric compatibility, missingness and finite values.

Verification:

- Phase 6 checksum and raw checksum remain unchanged.
- Every variant audit returns `PASS`.
- Matrix probes have shape `[5, F]` and preserve registered channel order.
- No data transformation or full-data fitting occurs.

### Step 3: Materialize and reload FEATURESETS-v1

Write the JSON source of truth, long CSV registry, variant summary, components, lineage, leakage audit, order checks, manifest, discrepancy log, human-readable specification and sign-off atomically. Write sign-off last and verify every output checksum.

Verification:

- Reloaded JSON arrays preserve order.
- Runtime fingerprints equal saved fingerprints.
- Baseline reference is `FS1_TF1` and all selection statuses remain `UNTESTED`.
- A second materialization verifies existing signed artifacts without rewriting them.

### Step 4: Extend automated tests and architecture

Add Phase 7 unit tests, extend integration from Phase 0-6 to Phase 0-7, document `data/feature_sets.py` as the sole Phase 7 owner and keep Phase 8 unstarted.

Verification:

- Unit and integration tests pass.
- `data/splitting.py` remains empty and no split artifact exists.

### Step 5: Add orchestration-only notebook section

Import and call `materialize_phase_7`, display the saved Phase 7 manifest and sign-off, and update the boundary heading to Phase 0-7. Do not place components, feature lists, hashes, matrix probes or audits in notebook code.

Verification:

- Phase headings are ordered through Phase 7.
- Phase 7 cell contains no definitions, loops, transformations or serialization.
- Notebook contains no machine-specific path, comments or icons.

### Step 6: End-to-end validation

Execute the notebook with a fresh kernel, validate every Phase 0-7 sign-off, collect and run the full test suite, verify raw and `FEATURES-v1` checksums, and inspect the final working-tree diff.

## Risks and controls

| Risk | Control |
|---|---|
| Order drift | Ordered tuples, deterministic construction and SHA-256 fingerprints |
| Metadata leakage | Explicit intersection audit against canonical metadata |
| Target leakage | Current feature registry contains no future target; `Appliances` role is historical-channel candidate only |
| Random-control contamination | Exact FS0/FS1/FS2 membership assertions |
| Silent extra columns | Explicit components; no dtype-driven or automatic column selection |
| Registry mutation | Internal tuples and defensive list copies |
| Partial artifact state | Atomic writes and sign-off written last |
| Upstream drift | Verify all Phase 6 signed output checksums before building Phase 7 |

## Alternatives rejected

- Reusing `data/features.py` would mix Phase 6 feature creation with Phase 7 variant selection.
- Using `data/splitting.py` would violate Phase 8 ownership.
- Defining lists in the notebook would create a second source of truth.
- Materializing six full CSV files would duplicate data and introduce drift.

## Completion criteria

The plan is complete only when all required artifacts reload correctly, all six variants and fingerprints pass audit, Phase 8 remains untouched, notebook execution is complete and the full test suite passes.
