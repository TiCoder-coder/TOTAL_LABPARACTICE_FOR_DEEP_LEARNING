# Model Improvement V2 — Step 16 Final-Refit / Lock Pre-Process Plan

## 1. Objective and scope

Prepare, but do not execute, the final three-model refit for the Human-selected
development policy `mean(E14-M1 seeds 42,123,2026)`. This contract is entirely
pre-Test. Step 17 and final benchmark evaluation are out of scope.

## 2. Final prediction policy

- Candidate/config source: `TR_C2_ALT_LOOKBACK_E14_M1`.
- Seeds: `42`, `123`, `2026`.
- Prediction: arithmetic mean of the three final-refit model predictions.
- Weights: `[1/3, 1/3, 1/3]`, immutable and not tunable.
- The nine rolling-origin Stage-B checkpoints remain development evidence.
  They are not final benchmark models and are not warm-start sources.

## 3. Final-refit epoch policy

Apply `MEDIAN_RO_INNER_BEST_EPOCHS-v1` to E14-M1 seed-42 Stage-A evidence:

| Fold | Stage-A run | `best_epoch_inner` |
|---|---|---:|
| RO1 | `RUN_V2_TR_E14_RO1_A_0001_B5DD2C2E` | 16 |
| RO2 | `RUN_V2_TR_E14_RO2_A_0006_EC2A3FCE` | 13 |
| RO3 | `RUN_V2_TR_E14_RO3_A_0011_DFA1252E` | 23 |

The median is 16. Phase45/46 forbids seed-specific epoch selection, so every
seed trains exactly 16 epochs. There is no validation, early stopping, manual
override, Test dependency, or post-E20 retuning.

## 4. Data and scaler policy

- Region: `V2_FINAL_DEV_REGION-v1`, exactly the locked L72/WB0 Train and
  Validation target population: `TGT_00000144` through `TGT_00016773`, 16,630
  ordered unique targets.
- Test rows and Test target IDs must never be loaded.
- Fit one X scaler and one Y scaler once on the 16,630 full-development target
  rows and reuse their immutable values across all three seeds.
- Follow `FINAL_SCALING-v1`: standardize the 28 continuous FS2_TF1 features;
  pass through `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, and `weekend`.
- Fit YS1 on raw `Appliances` Wh from the same full-development target rows.
- Preflight recomputes and verifies deterministic scaler statistics hashes in
  memory. Official execution writes the exact locked scaler payloads before
  the first model run.

## 5. Locked scientific configuration

FS2_TF1 (33 ordered features), L72, horizon 1, WB0, Transformer `d_model=64`,
4 heads, FFN 256, 2 Post-LN layers, last-step pooling, dropout 0.10, AdamW
LR `2e-4`, weight decay `1e-3`, batch 16, gradient clipping 1.0, scheduler OFF,
and hybrid level-plus-delta loss with `lambda_delta=0.10` and standardized-space
SmoothL1 `beta=1.0`.

Each seed starts from a fresh model and fresh optimizer. No model, optimizer,
or scheduler state is reused.

## 6. Output and checksum contract

All outputs belong under `artifacts/model_improvement_v2/final_model_lock/`.
Expected new runs: exactly three. Each run must preserve its immutable config
fingerprint, final scaler hashes, official epoch, training history/log, final
checkpoint SHA-256, lifecycle state, and `NOT_ACCESSED` Test status.

Checkpoint SHA-256 values are locked after successful training because they
cannot be precomputed without fabricating evidence. Scaler hashes and final
per-seed config fingerprints are locked before training in the config snapshot.

## 7. Recovery / no-retrain contract

- `official` requires an empty Step16 run namespace.
- A completed seed is reusable only after exact config, epoch, scaler,
  checkpoint SHA, history length, and Test-firewall verification.
- `resume-partial` may train only seeds lacking a completed canonical run.
- Completed seeds are never retrained or overwritten.
- Failed evidence is preserved; the next attempt receives a new attempt suffix.
- A RUNNING record is a hard stop pending a separate interruption audit.
- Final manifest creation requires exactly one verified completed run per seed.

## 8. Human gate and hard stops

Preparation/preflight does not authorize training. Stop on source-checksum,
config, population, feature-order, scaler, epoch-policy, ensemble-weight, or
Test-firewall drift; scheduler enablement; checkpoint warm-start; an unexpected
run; or missing explicit Human authorization.
