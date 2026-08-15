# Phase 09 Debug Cache Permission Fix Plan — 2026-08-12

## Objective

Resolve only the external cache-lock permission that prevented the approved
Phase 9 debug run from starting.

## Proposed Action

1. Request permission to run the existing offline debug command outside the
   filesystem sandbox restriction.
2. Continue using the current cached Rotten Tomatoes dataset and approved
   DistilBERT checkpoint; do not download or substitute data/model assets.
3. Execute the deterministic 1,000 Train / 200 Validation / one-epoch debug
   run.
4. Verify finite loss and metrics, real optimizer steps, a debug checkpoint,
   and `test_accessed=false`.
5. If debug fails for any implementation/runtime reason, stop again and update
   error analysis.
6. Only if debug passes, construct a fresh seeded model and start the approved
   full baseline.

## Files and Methodology

No Phase 0–8 file or methodology change is proposed. The Phase 9 source remains
the implementation under verification. Notebook Phase 9 presentation remains
pending a successful full run.

## Completion Criterion

The permission issue is resolved only when the real debug run completes and
`docs/result/phase_09_training/debug_verification.json` reports PASS.
