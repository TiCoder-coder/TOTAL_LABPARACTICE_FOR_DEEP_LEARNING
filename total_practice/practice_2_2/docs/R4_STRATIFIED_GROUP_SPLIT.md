# R4 Stratified Product and Visual Group Split

## Scope

R4 creates a deterministic 70/15/15 candidate split from the connected components produced by R3. It assigns whole components rather than individual images, preserves all original source files, excludes generated derivatives, creates no data loader, performs no model training, and does not evaluate Test.

The preferred lineage name is `v3_product_visual_group_s42_v1`. The current lineage is a cryptographically sealed candidate, not an authorized model-development lineage, because the R3 exit gate is blocked.

## Inputs

- R2 original-image inventory: `artifacts/new_work/r2_provenance_semantic_audit_v1/legacy_inventory.json`
- R3 components: `artifacts/new_work/r3_product_visual_groups_v1/group_manifest.json`
- R3 confirmed edges: `artifacts/new_work/r3_product_visual_groups_v1/accepted_edges.json`
- R3 review queue: `artifacts/new_work/r3_product_visual_groups_v1/review_queue.json`
- R3 gate report: `docs/R3_VERIFICATION.json`
- Split seed: `42`
- Eligible provisional images: 2,896
- Connected components: 2,462
- Generated images included: 0

Every R3 component contains one class label. R4 rejects overlapping components, duplicate component IDs, empty components, multi-label components, and any mismatch between component membership and the R2 original-image inventory.

## Assignment algorithm

Components are handled independently within each class. For each class, R4 computes target image counts and target component counts for Train, Validation, and Test. Components are ordered by descending size and then by a SHA-256 rank derived from seed 42 and the component ID.

Each component is assigned to the split that minimizes the combined squared fill ratio for images and components. A SHA-256 rank provides deterministic tie-breaking. This balances both observations and independent groups while preserving every component as one indivisible unit.

The complete assignment is recomputed after reversing the input component order. R4 aborts if the two assignment maps differ.

## Candidate distribution

| Split | Images | Image ratio | Components | Component ratio |
|---|---:|---:|---:|---:|
| Train | 2,034 | 70.2348% | 1,725 | 70.0650% |
| Validation | 431 | 14.8826% | 369 | 14.9878% |
| Test | 431 | 14.8826% | 368 | 14.9472% |

All ten classes appear in all three splits. The maximum deviations are:

- Overall image ratio: 0.2348 percentage points.
- Per-class image ratio: 0.4403 percentage points.
- Per-class component ratio: 0.3061 percentage points.

All values remain below the declared one-percentage-point limit.

## Leakage audit

R4 verifies split isolation separately for:

1. R3 connected-component ID.
2. Product ID when available.
3. Exact raw-file SHA-256.
4. Exact decoded-pixel SHA-256.
5. R2 source group.
6. Every R3 confirmed visual edge.

The candidate split has zero violations for all six strong relations. No confirmed component, identity relation, or accepted visual edge crosses a split boundary.

R4 also evaluates all unresolved R3 review edges against the candidate assignment. It documents 1,172 cross-split review-only pairs:

- Same-label uncertain pairs: 663.
- Cross-label uncertain pairs: 509.

These pairs are not represented as confirmed leakage, but they prevent the lineage from passing until review decisions are completed and R3 is rebuilt. Every pair is persisted with its deterministic edge ID, both split assignments, visual evidence, and review fields.

## Manifest authorization

The candidate manifest contains one row for each of the 2,896 provisional original images. Every row has a class, component, split, identity hashes, source group, product ID field, eligibility state, and model-use state.

Because the gate is blocked:

- `eligibility` is `provisional_pending_predecessor_gates`.
- `use_for_model` is `false` for every row.
- model training is not authorized.
- R5 has not been executed.

This prevents a structurally valid candidate split from being mistaken for an approved training input.

## Fingerprints

R4 records independent SHA-256 fingerprints for:

- the eligible dataset identity;
- R3 component membership;
- the asset-to-split assignment;
- the Test partition;
- the JSON manifest file;
- the CSV manifest file.

Two consecutive executions produced identical hashes for both manifests, the fingerprint file, leakage report, review-only cross-split list, and R4 verification report.

## Test guard

The Test partition is cryptographically sealed to the split and Test-partition fingerprints before any training. Its guard state is `sealed_candidate_blocked`.

The guard records:

- lineage authorization: false;
- model training authorization: false;
- Test access allowed: false;
- Test access count: 0;
- Test evaluation count: 0;
- authorization token hash: null;
- repeat evaluation allowed: false.

The guard API rejects access when the split fingerprint differs, the lineage is unauthorized, Test access is sealed, the authorization token is invalid, or Test has already been accessed.

## Artifacts

- `candidate_split_manifest.json`: structured asset-level candidate assignment.
- `candidate_split_manifest.csv`: tabular form of the same assignment.
- `split_summary.json`: image and component distributions overall and by class.
- `split_fingerprints.json`: logical and file-level fingerprints.
- `leakage_report.json`: confirmed isolation assertions and review-only counts.
- `review_only_cross_split_pairs.json`: every unresolved pair crossing the candidate boundary.
- `test_guard.json`: sealed Test access policy and counters.
- `split_report.json`: R4 gate result.
- `artifact_manifest.json`: size and SHA-256 for each persisted R4 output.
- `docs/R4_VERIFICATION.json`: machine-readable R4 verification report.

## Gate result

R4 remains blocked for three reasons:

1. R3 has not passed.
2. Product provenance coverage is zero, so unique-product distribution cannot be proven.
3. There are 1,172 unresolved review-only edges crossing candidate split boundaries.

Historical Validation and Test metrics are not comparable to this candidate lineage. No metric from the old canonical split is copied, relabeled, or presented as an R4 result.
