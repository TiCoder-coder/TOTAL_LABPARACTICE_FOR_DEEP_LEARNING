# Phase 6 Decode Sanity False Negative — 2026-08-12

## Error

The full Phase 0–6 notebook run stopped at the Phase 6 decode assertion.

## Context and Evidence

Tokenization completed for train, validation and test, and the inspected train
sample contained valid `input_ids`, `attention_mask` and `labels`. The decode
check compared sets created by plain whitespace splitting.

## Root Cause

Plain `str.split()` treats punctuation-attached forms such as `word,` and
`word` as different tokens. DistilBERT decoding can normalize spacing around
punctuation, causing a false-negative word-overlap score even when lexical
content is retained.

## Resolution

Normalize both original and decoded text to lowercase word tokens using the
same regular expression before calculating overlap. Keep the existing 0.8
acceptance threshold; do not weaken the criterion.

## Final Status

RESOLVED — all three executed decode checks achieved word overlap 1.0, and the
full Phase 0–6 notebook run completed without an error.
