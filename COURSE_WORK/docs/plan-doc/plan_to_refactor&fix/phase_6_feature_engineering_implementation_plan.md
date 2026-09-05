# Phase 6 Feature Engineering Implementation Plan

## Plan ID

`CW-PHASE-6-IMPLEMENT-001`

## Approval status

`APPROVED_BY_CURRENT_HUMAN_REQUEST`

## Objective

Implement the complete deterministic Phase 6 feature-engineering contract, materialize reproducible `FEATURES-v1` artifacts, and expose Phase 6 in `CourseWork.ipynb` through public API calls only.

## Inputs

- `DATA-v1`
- `SCHEMA-v1`
- `TEMPORAL-v1`
- `EDA-v1`
- `ENV-v1`
- `configs/base/coursework_contract.json`

## Outputs

- `data/interim/uci_appliances_energy_prediction/energydata_feature_engineered_v1.csv`
- `artifacts/features/feature_engineering_manifest.json`
- `artifacts/features/feature_registry.csv`
- `artifacts/features/feature_lineage.csv`
- `artifacts/features/feature_availability.csv`
- `artifacts/features/feature_leakage_audit.csv`
- `artifacts/features/feature_engineering_audit.csv`
- `artifacts/features/feature_engineering_discrepancies.json`
- `artifacts/features/feature_engineered_v1.sha256`
- `artifacts/features/phase_6_signoff.json`

## Impacted files

- `src/course_work/data/features.py`
- `tests/unit/test_features.py`
- `tests/integration/test_phase_0_to_5_chain.py`
- `tests/integration/test_notebook_boundary.py`
- `notebook_course_work/CourseWork.ipynb`
- `docs/RULE_BASE/architecture_rule.md`

## Implementation sequence

### Step 1: Implement pure Phase 6 transformations

Add constants and public functions that validate the temporal input, construct the canonical ordered base view, add the five cyclical/calendar features without mutating the input, build lineage and availability records, and validate invariants.

Verification:

- Synthetic formula tests for midnight, day boundary and weekend mapping.
- Determinism, input immutability, duplicate-column and invalid-timestamp tests.
- No imports or writes outside the approved dependency direction.

### Step 2: Implement artifact materialization and reload verification

Add `materialize_phase_6` and `load_validated_feature_view`. Verify upstream sign-offs and checksums, materialize all required CSV/JSON/checksum outputs atomically, reload them, calculate output checksums, and sign off only from completed audit results.

Verification:

- Raw CSV SHA-256 remains unchanged.
- Derived CSV row count and ordered columns match the contract.
- Timestamp, target, raw features and continuity segments match `TEMPORAL-v1`.
- All audit rows pass.
- Written checksum equals the derived file SHA-256.
- A second materialization verifies existing signed artifacts without rewriting them.

### Step 3: Extend automated tests through Phase 6

Add focused unit tests and replace the obsolete Phase 6 unstarted assertion with Phase 0-6 integration checks.

Verification:

- Unit tests pass.
- Phase-chain tests verify sign-off order, artifact checksums and upstream immutability.

### Step 4: Add orchestration-only notebook section

Import `materialize_phase_6`, append a Phase 6 Markdown section, call the public API, load only saved Phase 6 presentation artifacts through existing utilities or simple presentation reads consistent with the notebook contract, and display the manifest, audit and sign-off. No feature formulas or transformations are placed in notebook cells.

Verification:

- Notebook source contains `materialize_phase_6`.
- Phase headings are ordered through Phase 6.
- Non-Phase-5 notebook code has no processing definitions, loops, training or serialization.
- Notebook has no machine-specific path.

### Step 5: Align architecture documentation

Extend the active Phase-to-module mapping, artifact ownership, data lifecycle, test boundary and transition status from Phase 0-5 to Phase 0-6 without changing the existing architecture.

Verification:

- `data/features.py` is documented as the sole Phase 6 owner.
- Notebook responsibility remains public API call and display only.
- Phase 7 remains unstarted.

### Step 6: End-to-end validation

Execute Phase 6, reload every output, run the notebook from a clean kernel and run the complete test suite.

Verification:

- No notebook error output or stale execution count.
- All Phase 0-6 sign-offs remain valid.
- Full test suite passes.
- Raw source checksum remains unchanged.

## Risks and controls

| Risk | Control |
|---|---|
| Future leakage | Explicit availability and leakage audits; no shifted target or future covariates |
| Full-data fit leakage | Only row-local deterministic calendar transformations are allowed |
| Timestamp drift | Reuse `timestamp_parsed` and compare exact values/order |
| Raw mutation | Deep-copy inputs and compare raw/dataframe fingerprints and raw SHA-256 |
| Silent feature-order drift | Canonical ordered columns and explicit registry |
| Partial artifact state | Atomic writes and sign-off written last |
| Stale notebook state | Execute from a clean kernel and atomically replace only after success |

## Alternatives rejected

- Implementing feature formulas directly in the notebook violates the requested architecture.
- Adding a second Phase 6 source module duplicates ownership.
- Creating manual lag or rolling columns conflicts with the sequence-window design.
- Scaling or PCA before chronological split violates the leakage contract.

## Completion criteria

The plan is complete only when the source implementation, all required artifacts, tests, notebook orchestration, architecture documentation and clean-kernel execution satisfy the Phase 6 Definition of Done.
