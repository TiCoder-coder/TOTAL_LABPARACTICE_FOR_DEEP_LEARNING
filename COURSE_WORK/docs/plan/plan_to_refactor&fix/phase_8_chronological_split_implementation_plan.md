# Phase 8 Chronological Split Implementation Plan

## Plan ID

`CW-PHASE-8-IMPLEMENT-001`

## Approval status

`APPROVED_BY_CURRENT_HUMAN_REQUEST`

## Objective

Implement the fixed chronological 70/15/15 protocol in `data/splitting.py`, materialize reproducible `SPLIT-v1` membership and boundary artifacts, prepare WB0/WB1 handoff metadata, and enforce the Test firewall.

## Inputs

- `FEATURES-v1` derived master table and Phase 6 sign-off.
- `FEATURESETS-v1` registry and Phase 7 sign-off.
- `TEMPORAL-v1` manifest and Phase 4 sign-off.
- `configs/base/coursework_contract.json` split and boundary options.

## Outputs

- `artifacts/splits/split_manifest.json`
- `artifacts/splits/split_summary.csv`
- `artifacts/splits/split_membership.csv`
- `artifacts/splits/split_boundaries.csv`
- `artifacts/splits/split_boundary_neighborhood.csv`
- `artifacts/splits/split_leakage_audit.csv`
- `artifacts/splits/train_validation_distribution_summary.csv`
- `artifacts/splits/split_discrepancies.json`
- `artifacts/splits/figures/SPLIT_01_timeline.png`
- `artifacts/splits/phase_8_signoff.json`

## Impacted files

- `src/course_work/data/splitting.py`
- `tests/unit/test_splitting.py`
- `tests/integration/test_phase_0_to_5_chain.py`
- `tests/integration/test_notebook_boundary.py`
- `notebook_course_work/CourseWork.ipynb`
- `docs/RULE_BASE/architecture_rule.md`

## Implementation sequence

### Step 1: Implement pure chronological membership construction

Validate ratio sum, non-null sorted unique timestamps and required lineage columns. Calculate `train_end = floor(0.70*N)` and `validation_end = floor(0.85*N)`, assign canonical split IDs and zero-based within-split positions without mutating input data.

Verification:

- Counts are 13,814, 2,960 and 2,961 for current `FEATURES-v1`.
- Membership covers all 19,735 rows exactly once.
- Last Train timestamp is earlier than first Validation timestamp, and last Validation timestamp is earlier than first Test timestamp.
- Rebuilding twice produces identical membership.
- Invalid ratios, unsorted timestamps, duplicate timestamps and null timestamps fail explicitly.

### Step 2: Implement fingerprints, boundary and Test-firewall audits

Compute ordered per-split fingerprints from timestamp, raw row index and split ID, then compute a global fingerprint from ratios, boundaries and ordered membership. Record boundary segments and a deterministic plus/minus five-row neighborhood. Audit coverage, disjointness, chronology, lack of randomization, shared split semantics, target-timestamp assignment and Test lock.

Verification:

- Every audit row is `PASS`.
- Fingerprints are deterministic and differ across split identities.
- Boundary neighborhood contains only structural metadata.
- No Test target or feature value is exported into diagnostic artifacts.

### Step 3: Create permitted Train/Validation diagnostics and structural figure

Generate raw-scale descriptive summaries for the pre-registered variables `Appliances`, `lights`, `T1`, `RH_1`, `T_out` and `RH_out` using Train and Validation only. Create a timestamp-only horizontal split timeline.

Verification:

- Distribution summary contains only `TRAIN` and `VALIDATION`.
- No Test distribution statistic is calculated or saved.
- Timeline contains periods, row counts and boundaries but no target values.

### Step 4: Materialize and reload SPLIT-v1

Write membership, summaries, boundaries, neighborhood, leakage audit, distribution summary, figure, discrepancy log, manifest and sign-off atomically. Write sign-off last and verify every output checksum.

Verification:

- Reloaded membership schema, counts, ordering and global fingerprint match the manifest.
- A second materialization validates existing signed artifacts without rewriting them.
- Raw and `FEATURES-v1` checksums remain unchanged.

### Step 5: Extend tests and architecture

Add Phase 8 unit tests, extend Phase integration through `SPLIT-v1`, document `data/splitting.py` as the sole Phase 8 owner and leave Phase 9 `data/scaling.py` empty.

Verification:

- Targeted unit and integration suites pass.
- No scaling artifact exists.

### Step 6: Add orchestration-only notebook section

Import and call `materialize_phase_8`, display the saved split manifest and sign-off, and update the terminal boundary to Phase 0-8. No split calculations, DataFrame slicing, distributions, plotting or serialization may appear in Phase 8 notebook code.

Verification:

- Phase headings are ordered through Phase 8.
- Phase 8 code contains no definitions, loops, randomization, transformations or file writes.
- Notebook contains no machine-specific path, comments or icons.

### Step 7: End-to-end validation

Execute the notebook with a fresh kernel, verify every signed Phase 0-8 checksum, collect and run the complete test suite, verify the Test firewall and inspect the final working-tree diff.

## Risks and controls

| Risk | Control |
|---|---|
| Temporal leakage | Strict chronological order and target-timestamp membership contract |
| Boundary ambiguity | Half-open ranges and explicit start/end positions |
| Test leakage | Structural-only Test metadata and `LOCKED_UNTIL_PHASE_47` status |
| Split drift | Per-split and global SHA-256 fingerprints |
| Variant-specific samples | One membership registry shared by all `FEATURESETS-v1` variants |
| Lost WB0 context | Preserve one full master table plus row membership instead of isolated data copies |
| Partial artifacts | Atomic writes and sign-off written last |
| Upstream drift | Verify Phase 6 and Phase 7 signed checksums before split construction |

## Alternatives rejected

- Random or stratified splitting violates forecasting chronology.
- Splitting windows instead of target periods risks overlap and leakage.
- Creating three full feature CSVs duplicates data and can destroy WB0 context.
- Inspecting Test distributions before final lock violates the Test firewall.

## Completion criteria

The plan is complete only when `SPLIT-v1` reloads deterministically, every split and firewall audit passes, Phase 9 remains untouched, notebook execution is complete and the full test suite passes.
