# Part 2.2 — Protocol Infrastructure Refactor Plan

## Objective

Implement the frozen `practice_3_v2.0` infrastructure without constructing/running a real experiment, accessing v1 Test, evaluating v2 Holdout, selecting a real winner, or modifying v1 evidence.

## Files to add

- `processing_own_phase/experiment_protocol_v2.py`
- `processing_own_phase/dataset_protocol_v2.py`
- `processing_own_phase/experiment_registry_v2.py`
- `processing_own_phase/holdout_guard_v2.py`
- `processing_own_phase/experiment_runner_v2.py`
- `tests/test_protocol_v2.py`
- real protocol metadata under `docs/result/practice_3_v2/`
- `docs/save_process_proceduce_own_phase_refactor&fix/part_02_2_protocol_infrastructure_log_2026-08-15.md`

## Files to modify

None of the v1 source, notebook, or historical result files are planned for modification. The frozen Part 2.1 plan is read-only.

## v1 compatibility

- v2 writes only to `docs/result/practice_3_v2/` and `runs/practice_3_v2/`.
- Official v1 Test is never loaded.
- Phase 9/11/14 artifacts and model/package hashes are verified before and after.
- Existing Part 1 working-tree changes are preserved and not folded into v2 behavior.

## Artifact namespace

The real namespace contains protocol manifest, split manifest, sealed holdout state, three PLANNED registry records, and three immutable run configs. It contains no metric, checkpoint, winner, training history, or synthetic evidence.

## Synthetic-test strategy

Use temporary directories and in-memory balanced records to verify deterministic splitting, counts, leakage failures, canonical config hashes, only-LR differences, registry protections/lifecycle, ranking tolerance/ties, winner validation, and one-time holdout state transition. No synthetic winner/metrics are written to real results.

## Optimizer compatibility preflight

Instantiate a zero-step `torch.optim.AdamW(..., fused=True)` with a tiny parameter on the selected execution device. Do not compute loss, backward, or optimizer step. Record only `SUPPORTED` or `BLOCKED` plus device/version metadata. If blocked, create the required analysis-error document and do not claim readiness.

## Holdout safety

Dataset construction reads only official Train and Validation. The manifest may contain source IDs, counts, hashes, and leakage status, but never holdout text. Holdout materialization/evaluation is not implemented or called. The guard begins sealed with count zero and is exercised only with synthetic state under a temporary directory.

## Risks and rollback

- Dataset cache unavailable: stop dataset finalization; do not substitute data.
- Exact duplicates/conflicting labels: stop and write analysis error; do not alter counts.
- Fused optimizer unavailable: document blocker; do not fall back.
- Partial metadata write: use canonical atomic writes and validate read-back.
- Accidental historical mutation: compare recorded hashes/counts and Git blobs.
- Protocol drift: validate the Part 2.1 plan SHA-256 in the protocol manifest.

Rollback consists only of removing newly created v2 files; no v1 file needs restoration.

## Definition of done

All real metadata matches the frozen protocol; the development pool excludes official Test; split counts/hashes/leakage checks pass; three run configs differ only by LR; optimizer preflight is supported; registry/TensorBoard/training-interface/ranking/winner/holdout infrastructure passes static and synthetic tests; v2 holdout count remains zero; v1 Test count/hash remain unchanged; and no Trainer, training, Test/Holdout evaluation, or real winner occurs.

