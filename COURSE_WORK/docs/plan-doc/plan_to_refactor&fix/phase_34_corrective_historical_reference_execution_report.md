# Phase 34 Corrective Historical Reference Execution Report

## Final scope

Phase 34 — S12 Head Sweep is finalized from the Human-authorized H2 run and the approved H4 historical reference. H2 was not rerun during finalization, H4 was not retrained, Test remained forbidden, and Phase 35 was not executed.

## H2 completed run

```text
condition=H2
run_id=RUN_TR_S12_0020_DE823D66
config_fingerprint=de823d669a5b8f395e2a01eb85ce66daa347641438003b3c7627b9a196e26525
d_model=64
num_heads=2
head_dim=32
num_layers=2
ffn_dim=128
seed=42
epochs_executed=22
best_epoch=12
validation_rmse_wh=58.65958891509437
validation_mae_wh=26.345127858930915
validation_r2=0.5955657515395012
validation_samples=2960
population_fingerprint=a40ded8802e90008535d268720bad9e9dcca5eee1436ddb359deea3ad39a1987
test_access=FORBIDDEN
```

The registry record, config, status, best checkpoint, training history, Validation metrics, finite Validation predictions, population/sample alignment and terminal evidence are consistent. Early stopping completed the run after epoch 22; the canonical best checkpoint is epoch 12.

## H4 historical reference

```text
condition=H4
run_id=RUN_TR_S09_0016_AE0FB819
config_fingerprint=ae0fb819cd0ecca797892a5f6e2f9758c7809c01f5ab92daaeca21a880d5a6ba
d_model=64
num_heads=4
head_dim=16
num_layers=2
ffn_dim=128
best_epoch=15
validation_rmse_wh=58.08190056355405
validation_mae_wh=27.595002038670813
validation_r2=0.6034923842647237
evidence_mode=HISTORICAL_REFERENCE_WITH_INCOMPLETE_ARTIFACT_RETENTION
evidence_status=PASS_WITH_WARNING
test_access=FORBIDDEN
```

Phase 33 sign-off, winner, reference update, Experiment Registry record, H4 config, status and best Validation metrics agree. The original H4 checkpoint, training history, Validation predictions and training log remain explicitly unavailable and were not fabricated.

## Full-precision selection

| Condition | Heads | Head dimension | Validation RMSE Wh | Evidence |
|---|---:|---:|---:|---|
| H2 | 2 | 32 | 58.65958891509437 | Complete run artifacts |
| H4 | 4 | 16 | 58.08190056355405 | Historical reference, incomplete retention |

The canonical rule minimizes full-precision Validation RMSE Wh; exact ties prefer H2. H4 therefore wins with a margin of `0.5776883515403215 Wh`.

## Final result and handoff

```text
winner=H4
winner_run_id=RUN_TR_S09_0016_AE0FB819
selected_num_heads=4
selected_head_dim=16
phase_34_status=PASS_WITH_WARNING
warning=H4_SOURCE_ARTIFACT_RETENTION_INCOMPLETE
approved_for_phase35=true
test_access=FORBIDDEN
```

Canonical Phase 34 evidence is stored under `artifacts/sweeps/S12_heads/`. The Phase 35 handoff propagates the historical-retention warning. Approval is recorded as data only; Phase 35 has not been executed.

## Logs

```text
Terminal execution log:
artifacts/sweeps/logs/phase_34_h2_RUN_TR_S12_0020_DE823D66_terminal.log

Machine-readable processing log:
docs/save_log_in_processing/phase_34_s12_head_log.json
```
