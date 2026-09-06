# Phase 34 full pre-training dependency recovery

Date: 2026-08-24 (Asia/Ho_Chi_Minh)

## Scope

The audited runtime path is `scripts/run_single_condition.py S12_HEADS H2` through the call boundary immediately before `TrainingEngine.train()`.

The direct signed dependency set is Phase 5, 7, 8, 9, 10, 11, 12, 13 immutable registry metadata, and Phase 33 handoff evidence. Phase 1-4 and Phase 6 scientific tables are not materialized by this runtime path and were not regenerated.

## Root cause

Thirty-three signed CSV/data-view artifacts required by the pre-training path were absent. The affected file classes are ignored by repository rules, while their canonical sign-offs and expected SHA256 values remain available. This caused sequential checksum failures even though the preserved JSON manifests were valid.

## Recovery result

All 33 invalid direct-dependency artifacts were rebuilt only in a temporary project copy with the canonical Phase 5/7/8/10/11 generators. Phase 13 `experiment_families.csv` was rebuilt with the canonical registry serializer. A candidate was copied to the canonical path only after its SHA256 exactly matched the signed checksum.

- Signed artifacts checked: 85
- Exact pre-existing PASS: 52
- Exact reconstructed: 33
- Still invalid on the direct path: 0
- Phase 9/10/11/12 materialization: PASS
- `load_upstream_context()`: PASS
- target scaler: PASS
- Train/Validation loaders: PASS
- Phase 34 preflight: PASS
- H4: `HISTORICAL_REFERENCE_WITH_INCOMPLETE_ARTIFACT_RETENTION`, `PASS_WITH_WARNING`
- H2: the only `TRAIN_NEW` condition
- Test evaluation/access: FORBIDDEN

The full Phase 13 historical registry validator still reports pre-existing missing retained artifacts for older completed runs. That broad historical-retention audit is not called by the H2 launch path. The actual Phase 13 prerequisites used by H2 (canonical JSONL loading, family contract, config validation, and registration/start behavior) passed in the guarded temporary smoke invocation.

## Validation notes

The guarded smoke invocation used a temporary clone and replaced only `TrainingEngine.train()` in memory with a stop exception. It reached that exact boundary after config construction, upstream loading, target-scaler validation, Train/Validation loader construction, registry registration/start, and model construction. No epoch ran and no H2 artifact was written to the canonical workspace.

Focused tests passed: `98 passed`. Six additional `test_splitting.py` setup cases remain outside this direct-path recovery because they invoke Phase 2 full materialization and fail on the unrelated missing `data/raw_data/README_SOURCE.md`; Phase 2 was not regenerated.
