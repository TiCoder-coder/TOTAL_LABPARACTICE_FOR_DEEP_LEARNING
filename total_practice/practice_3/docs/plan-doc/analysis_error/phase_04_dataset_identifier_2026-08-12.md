# Phase 4 Dataset Identifier Error — 2026-08-12

## Error

`HfUriError`: the short dataset identifier `rotten_tomatoes` was rejected by
the installed Hugging Face stack because the generated dataset URI did not
contain the required namespace.

## Phase

Phase 4 — Dataset Loading, during Priority 2 EDA verification.

## Context and Evidence

The executed `load_dataset("rotten_tomatoes")` call failed before any EDA was
performed. The overall plan specifies the canonical identifier
`cornell-movie-review-data/rotten_tomatoes`.

## Root Cause

The implementation used an abbreviated identifier that is not accepted by the
installed `datasets==5.0.1` / Hugging Face Hub URI parser.

## Resolution

Use the canonical identifier from `plan.md` as the single `DATASET_NAME`
constant and load the dataset through that constant.

## Final Status

RESOLVED — the canonical identifier loaded all three official splits and the
Phase 4 dataset contract passed.
