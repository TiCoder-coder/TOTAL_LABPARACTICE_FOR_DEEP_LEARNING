# Phase 34 full pre-training dependency recovery plan

Status: COMPLETED

## Executed corrective sequence

1. Trace `run_condition(S12_HEADS, H2)` to the `TrainingEngine.train()` boundary.
2. Audit signed outputs for the direct dependency phases in one pass.
3. Clone the project to a temporary path.
4. Rebuild Phase 5, 7, 8, 10 and 11 outputs with their canonical generators; rebuild Phase 13 family metadata with its canonical serializer.
5. Compare every candidate to its signed SHA256 and restore exact matches only.
6. Verify Phase 9-12 materializers, upstream context, target scaler, Train/Validation loaders, Phase 34 preflight, audit-only and dry-run.
7. Run a guarded H2 smoke invocation in the temporary clone and stop at `TrainingEngine.train()`.
8. Run focused tests and stop before H2 training.

## Preserved scientific protocol

- H2 is `TRAIN_NEW`, seed 42, Validation only.
- H4 is not retrained and remains `PASS_WITH_WARNING` historical evidence.
- No Test evaluation or Test target artifact was produced.
- No sign-off/checksum, experiment result, checkpoint, or Phase 34 protocol was modified.

## Next action

The human may run the documented H2 terminal command. Automated recovery must not execute it.
