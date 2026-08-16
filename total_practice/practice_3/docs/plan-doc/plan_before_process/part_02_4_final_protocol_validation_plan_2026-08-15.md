# Part 2.4 — Final Protocol Validation Plan

## Objective

Close Part 2 through a read-only final audit, documentation consolidation and execution-freeze verification. No model, tokenizer, Dataset, Trainer, optimizer or Holdout provider is constructed or called.

## Dependencies

- Part 2.1 frozen protocol design.
- Part 2.2 infrastructure plan/log and real NOT_STARTED metadata.
- Part 2.3 plan/log and PASS pretraining readiness manifest.
- v1 Phase 9/11/14 historical evidence.

## Frozen state

Expected protocol `practice_3_v2.0`, execution hash `fdfcbb87b20d0bc618890a51c70a9689d618682b5a7cdf8edf05b638fa347055`, three PLANNED runs, no metrics/winner/events, SEALED Holdout count zero, and v1 Test count one.

## Audit checks

- Recompute execution/readiness/component hashes without regeneration.
- Validate all locked protocol values and three run config hashes.
- Validate metadata-only dataset counts/classes/duplicates/overlaps.
- Validate registry result fields remain null and paths isolated.
- Validate optimizer/Early Stopping/TensorBoard/ranking/winner/cache contracts from stored evidence and source.
- Verify Holdout remains sealed/unmaterialized and official Test was not loaded.
- Verify v1 checkpoint/package hashes, Test manifest, and notebook Git blob remain unchanged.
- Verify Git changes remain under Practice 3.

## Documentation consolidation

Create:

- `docs/result/practice_3_v2/PART_02_PROTOCOL_README.md`
- `docs/result/practice_3_v2/part_02_final_audit.json`
- `docs/result/practice_3_v2/part_02_completion_report.md`
- Part 2.4 process log.

All status/values must be derived from frozen manifests; no training metric is invented.

## Execution-freeze verification

Hash exactly the Part 2.3 freeze components: Part 2.1 plan hash, protocol manifest hash, split manifest hash, three run config hashes and ranking representation. A mismatch is protocol drift and blocks Part 3; the hash is never regenerated to conceal drift.

## Prohibited operations

No model/tokenizer/dataset/Trainer construction; no train/evaluate/predict; no backward/optimizer/scheduler step; no v1 Test access; no v2 Holdout materialization/evaluation; no ranking/winner; no registry transition; no frozen protocol or historical artifact edit.

## Validation strategy

Static JSON/schema/hash/path/state checks, source audit, existing lightweight synthetic tests, Git blob/hash comparison and documentation link checks. Do not rerun expensive Part 2.3 integration.

## Readiness criteria

All hashes and locked values match; readiness remains PASS; v2 has no result evidence/winner/events; Holdout is sealed at count zero; v1 is unchanged; documentation is complete; and final audit records `ready_for_part_3=true`.

## Definition of done

Part 2 is complete only when Part 2.1–2.4 are PASS, protocol drift is false, execution hash is unchanged, documentation is consolidated, no prohibited operation occurred and final readiness is YES.

