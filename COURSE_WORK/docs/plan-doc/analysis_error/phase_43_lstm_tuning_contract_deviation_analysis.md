# Phase 43 — LSTM Tuning Scientific Contract Deviation Analysis

## Scope

This document records the confirmed mismatch between the authoritative Phase 43 LSTM tuning plan and the currently uploaded Phase 43 implementation/artifacts.

## Authoritative contract

The Phase 43 plan requires:

- standard unidirectional LSTM (`LSTM_IMPL-v1`);
- shared Phase 42 data/task context;
- five sequential one-factor stages: LT1 hidden size, LT2 layers, LT3 dropout when applicable, LT4 learning rate, LT5 weight decay;
- `max_epochs = 50`;
- `patience = 10`;
- `min_delta = 0`;
- gradient clipping max-norm 1.0;
- AdamW;
- MSE;
- fixed seed 42;
- fixed shared Train/Validation populations;
- full Validation evaluation each epoch;
- BEST selection by Validation RMSE Wh;
- no Test access;
- maximum 10 fresh scientific runs;
- final tuned run must be a verified completed run;
- Phase 44 handoff is allowed only after a valid Phase 43 signoff.

## Confirmed implementation deviation

The uploaded `phase43_lstm_tuning.py` explicitly overwrites the training budget:

```python
base_config["training"]["max_epochs"] = 2
base_config["training"]["patience"] = 2
```

This contradicts the Phase 43 hard contract requiring E50/P10.

The script also creates runtime DataLoaders with batch size 1024 while the stored run config records batch size 64. This creates an execution/config provenance inconsistency unless the engine subsequently reconstructs the configured loader, which the uploaded script does not show: the `train_loader` and `val_loader` constructed with batch size 1024 are passed directly to `engine.train()`.

## Artifact evidence

The current tuned winner records:

- run ID: `RUN_LS_LST_0049_7B728C6B`;
- `best_epoch = 2`;
- `max_epochs = 2`;
- `patience = 2`;
- Validation RMSE = `76.58898602123719 Wh`.

Therefore the winner is a product of the truncated implementation, not the authoritative E50/P10 protocol.

The current Phase 43 signoff states `PASS` and `ready_for_phase44 = true`, but those declarations are inconsistent with the plan because the tuning budget itself drifted.

## Additional provenance concerns

The uploaded signoff records `shared_lookback = 36`, while the uploaded tuned winner's config records `lookback_steps = 144`. This must be reconciled against the canonical Phase 42 handoff before any new scientific training.

The signoff records LT-stage winners that do not match the uploaded final winner config in several fields. For example, the signoff states:

- LT1 hidden size = 64;
- LT2 num layers = 2;
- LT4 learning rate = 3e-4;
- LT5 weight decay = 0.001.

The uploaded final winner config instead contains:

- hidden size = 128;
- num layers = 1;
- learning rate = 1e-3;
- weight decay = 1e-4.

This is a serious lineage inconsistency and must be resolved from the actual stage artifacts/run registry rather than by trusting either summary.

## Scientific impact

The current Phase 43 tuned LSTM cannot be treated as canonical Phase 43 evidence because:

1. the training budget is not the approved budget;
2. the current winner was selected after truncated training;
3. the signoff claims PASS despite the budget violation;
4. handoff to Phase 44 was released from an invalid tuning result;
5. the uploaded signoff and winner disagree on shared lookback and tuned hyperparameters.

## Required disposition

- Preserve current runs/artifacts as historical invalidated evidence.
- Do not delete or overwrite historical model bytes.
- Do not reuse the current tuned winner as the Phase 44 canonical LSTM.
- Reconcile the canonical Phase 42 handoff before rebuilding Phase 43.
- Correct the Phase 43 implementation first.
- Run a non-training preflight/dry-run proving E50/P10 and exact shared-data parity.
- Human performs official Phase 43 training manually.
- Verify Phase 43 PASS before any Phase 44 execution.
- Test remains locked throughout.
