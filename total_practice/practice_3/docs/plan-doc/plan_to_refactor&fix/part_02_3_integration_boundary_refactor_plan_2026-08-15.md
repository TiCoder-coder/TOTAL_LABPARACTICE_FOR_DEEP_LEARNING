# Part 2.3 — Integration Boundary Refactor Plan

## Changes

- Extend `dataset_protocol_v2.py` with real Train/Validation-only materialization from frozen source IDs.
- Extend `experiment_runner_v2.py` with dry Trainer construction and fused-optimizer creation helpers.
- Add `pretraining_integration_v2.py` for the zero-training review and readiness manifest.
- Extend tests with provider-denial and integration-contract checks.

## Safety

- The Dataset selector rejects Holdout requests and never indexes Holdout source IDs.
- The Trainer factory accepts exactly two datasets and has no Holdout/Test argument.
- No function in the Part 2.3 execution path calls train, backward, optimizer step, scheduler step, evaluate, or predict.
- Real registry/state remain read-only.

## Validation

Real cached Train/Validation, cached tokenizer/model, two dry batches, fresh model/Trainer/optimizer identity for all three configs, forward-only finite logits/loss, callback/config checks, and v1/hash guards.

## Rollback risk

Only newly added integration helpers/readiness evidence are affected. v1 and frozen Part 2.1/2.2 metadata are not rewritten.

## Completion

Integration review passes and readiness manifest freezes the unchanged execution protocol, or stops without readiness if any check fails.

