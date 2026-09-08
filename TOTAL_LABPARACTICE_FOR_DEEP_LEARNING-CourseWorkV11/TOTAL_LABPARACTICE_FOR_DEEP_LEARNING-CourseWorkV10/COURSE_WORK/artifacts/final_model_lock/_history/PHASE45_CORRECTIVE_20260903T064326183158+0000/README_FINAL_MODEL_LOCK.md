# Final Model Lock (Phase45)

**Immutable lock package for Phase 46 three-seed final refits.**

Phase 45 is a NO-TRAIN governance phase. It reads Phase 44 evidence and
writes the canonical lock package to `artifacts/final_model_lock/`.

## Contents

- `phase_45_signoff.json` — final PASS/FAIL/WARN
- `phase46_three_seed_handoff.json` — what Phase46 must consume
- `phase47_test_evaluation_guard.json` — Test-access guard
- `final_model_lock_fingerprint.json` — SHA256 over (config, recipe, lineage)
- `final_epoch_policy.json` — median(RO1,RO2,RO3) freeze
- `final_data_region_contract.json` — FINAL_DEV_REGION-v1 contract
- `final_scaling_contract.json` — FINAL_SCALING-v1 contract
- `final_model_scientific_config.json` — locked scientific config
- `final_three_seed_run_matrix.csv` — planned Phase46 run matrix
- See O45.* filenames for the 38 canonical artifacts.

## Failure policy

Any modification of these artifacts after a PASS signoff requires:
  - Protocol Amendment + new lock version.

## Reproducibility

All four fingerprints are computed from canonical deterministic serialization
(`course_work.utils.artifacts.canonical_json_bytes`). No timestamps in inputs.
