# Part 2.3 Integration Gap Analysis

## Context

Part 2.2 correctly created split metadata and a non-executing runner contract. Part 2.3 now requires real zero-training Dataset/Trainer integration.

## Gap 1 — dataset boundary

`load_development_pool()` iterates every original Train/Validation row to create the Part 2.2 manifest. Reusing it in Part 2.3 would iterate records assigned to v2 Holdout, violating the stricter requirement that Holdout not be materialized or inspected during integration review.

## Gap 2 — dry Trainer assembly

The Part 2.2 runner builds TrainingArguments/callbacks but intentionally has no Trainer-construction helper. Part 2.3 needs a safe construction path that accepts only explicit v2 Train/Validation datasets and never calls training.

## Required fix

- Add a manifest-driven Train/Validation selector that never indexes Holdout IDs.
- Add a dry Trainer factory and optimizer-construction verification path with no step.
- Add a dedicated Part 2.3 integration module/test; do not alter protocol values.

## Methodology impact

None. This enforces the existing access boundary and frozen configuration.

## Status

CONFIRMED INTEGRATION GAP — safe refactor required before review execution.

