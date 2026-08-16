# Part 3.1 — Pre-run Execution Verification Plan

## Objective

Perform the final read-only authorization gate before Part 3.2 may implement/execute the three real controlled training runs.

## Checks

- Recompute and compare the frozen execution protocol hash.
- Validate protocol/config/split/readiness/final-audit hashes and values.
- Confirm three unique PLANNED runs, only-LR experimental difference and no real results.
- Confirm Train/Validation/Holdout metadata boundary and sealed Holdout state.
- Check current device, fused AdamW construction support, TensorBoard dependency, directory writability and worst-case disk capacity.
- Check isolated/clean output, checkpoint, history, summary and TensorBoard paths.
- Audit the future single-run lifecycle/fresh-state contract and fixed execution order.
- Reject a pre-lock Holdout request before provider invocation.
- Verify v1 Test count and checkpoint/package hashes.

## Prohibited operations

No Dataset/model/tokenizer/Trainer construction; no training/evaluation/prediction; no backward or optimizer/scheduler step; no official Test access; no Holdout materialization; no real registry transition/ranking/winner; no protocol or historical-artifact edit.

## Validation

Static JSON/schema/hash/source/path checks plus zero-step optimizer construction, filesystem writability probes in temporary files, disk-usage calculation and the existing lightweight synthetic tests. No Part 2.3 integration rerun.

## Definition of done

Create `part_03_1_training_authorization.json` with `training_authorized=true` only if every critical check passes. Otherwise document the blocker, set authorization false and stop before Part 3.2.

