# MODEL_IMPROVEMENT-v2 E20 Pre-Process Plan

## 1. Authority and current state

This plan follows `docs/Plan_improve_model.md`, the Model Improvement V2 pre-process plans, and Human decisions through E18.

```text
FINALIST_SELECTION_STATUS = PASS
WAVE2_FINALIST = TR_C2_ALT_LOOKBACK_E14_M1
E19_STATUS = NOT_TRIGGERED
E20_TRAINING_AUTHORIZED = NO
TEST_STATUS = NOT_ACCESSED
```

E10, E11, E12, E15, and E16 changes are rejected and must not be inherited. E17 and fine-tuning are deferred. E19 has no Human-approved trigger and must not be opened by E20 work.

## 2. Finalist evidence

`TR_C2_ALT_LOOKBACK_E14_M1` is the latest Human-promoted incumbent. Its accepted development evidence is:

- pooled RMSE `58.5534531302616 Wh`;
- pooled MAE `26.33779907425144 Wh`;
- pooled R2 `0.5970279545320498`;
- worst-fold RMSE `65.37273386173845 Wh`;
- fold-RMSE SD `5.965764759957666 Wh`;
- provenance, population, fold-local scaling, leakage, and Test-firewall gates PASS.

No later candidate was promoted. E14-M1 is therefore eligible to enter E20 as the single Wave 2 finalist.

## 3. Scientific question

Does the Human-selected E14-M1 finalist retain its development advantage over the locked E01-equivalent control across matched seeds `42`, `123`, and `2026`, rather than benefiting from seed 42?

E20 is seed confirmation of two already locked configurations. It must not tune either configuration or select a best seed.

## 4. Locked configurations

### Control

```text
candidate = TR_C2_ALT_LOOKBACK
source = E01 exact configuration
config_fingerprint = 585c5e79e6a1c8c49efdd2f7767e6d3d4c628835acfda360a2fa66a2ef5c1e24
prediction = DIRECT
features = FS2_TF1 (33)
lookback = 72
horizon = 1
boundary = WB0_CONTEXT_CARRY_OVER
optimizer = AdamW
learning_rate = 3e-4
batch_size = 32
weight_decay = 1e-3
loss = MSE
scheduler = OFF
```

### Finalist

```text
candidate = TR_C2_ALT_LOOKBACK_E14_M1
config_fingerprint = ff99dba3b57d38308f412a76c936b3a1f2a99ab8835e465417090ebfda49bc77
prediction = DIRECT
features = FS2_TF1 (33)
lookback = 72
horizon = 1
boundary = WB0_CONTEXT_CARRY_OVER
d_model = 64
num_heads = 4
ffn_dim = 256
num_layers = 2
norm_order = POST_LN
optimizer = AdamW
learning_rate = 2e-4
batch_size = 16
dropout = 0.10
gradient_clip_max_norm = 1.0
weight_decay = 1e-3
loss_policy = HYBRID_LEVEL_PLUS_DELTA
lambda_delta = 0.10
scheduler = OFF
```

Both configurations retain their own already accepted scientific settings. E20 changes only the replication seed within each locked configuration.

## 5. Matched seed and fold protocol

- Seeds: exactly `[42, 123, 2026]`.
- Folds: exactly RO1, RO2, RO3.
- Each configuration/seed/fold requires Stage A and Stage B.
- Stage A uses fresh model, optimizer, scheduler state, fold-local X/Y scalers, and the locked selection protocol.
- Stage B uses a fresh model and optimizer, no warm start, no validation, no early stopping, and refits for the exact Stage-A best epoch from the same configuration/seed/fold.
- DataLoader seed/order policy must be identical between paired configurations for each seed.
- Ordered target IDs and evaluation populations must be identical across configurations and seeds.
- Test remains inaccessible.

## 6. Seed-42 reuse lock

The existing seed-42 evidence was audited at path, registry status, config, population, feature, Test authorization, artifact existence, and SHA-256 levels.

### E01 control — reuse six completed runs

```text
RO1_A RUN_V2_TR_E01_RO1_A_0001_3B5C1B75
RO2_A RUN_V2_TR_E01_RO2_A_0002_1EA6841D
RO3_A RUN_V2_TR_E01_RO3_A_0003_4C4B28D5
RO1_B RUN_V2_TR_E01_RO1_B_0004_533B5F33
RO2_B RUN_V2_TR_E01_RO2_B_0005_19A72D17
RO3_B RUN_V2_TR_E01_RO3_B_0006_AA5D66A6
```

### E14-M1 finalist — reuse six completed runs

```text
RO1_A RUN_V2_TR_E14_RO1_A_0001_B5DD2C2E
RO2_A RUN_V2_TR_E14_RO2_A_0006_EC2A3FCE
RO3_A RUN_V2_TR_E14_RO3_A_0011_DFA1252E
RO1_B RUN_V2_TR_E14_RO1_B_0016_668187E3
RO2_B RUN_V2_TR_E14_RO2_B_0021_AD87B689
RO3_B RUN_V2_TR_E14_RO3_B_0026_ABCFB6CD
```

```text
SEED42_REUSE_STATUS = PASS
SEED42_REUSED_RUNS = 12
SEED42_RETRAIN = FORBIDDEN
```

Implementation preflight must revalidate the immutable run IDs, candidate identities, seed, fold/stage mapping, config fingerprints, population/fold fingerprints, feature order, registered checkpoint/config SHA-256, and `test_access_authorized=false`. Any mismatch disables reuse and is a hard stop; it does not authorize automatic retraining.

## 7. Evidence and compute scope

```text
E20_EXPECTED_TOTAL_EVIDENCE = 36
E20_REUSED_SEED42_RUNS = 12
E20_EXPECTED_NEW_RUNS = 24
```

The 24 new runs are exactly:

```text
2 locked configurations x 2 new seeds x 3 folds x Stage A/B
```

No seed-42 run may be duplicated. No additional candidate or seed may be added.

## 8. Implementation and preflight preparation

Implementation must be isolated to the E20 V2 namespace and provide:

1. immutable E20 configuration and finalist-selection snapshot;
2. explicit candidate/seed matrix and expected canonical identities;
3. read-only cross-namespace seed-42 reuse ledger with checksum validation;
4. fresh-state Stage A/B execution for seeds 123 and 2026 only;
5. partial recovery that reuses verified completed canonical runs and trains only genuinely missing runs after Human authorization;
6. finalization-only mode when all canonical training evidence exists;
7. Test firewall and official-mode Human authorization gate;
8. focused tests and preflight proving no training is executed.

Preflight must report at least finalist/control identities, seeds, folds, total/reused/new run counts, seed-42 checksum/config/population status, feature/scaler/data-order equivalence, Test rows/targets seen, and `training_executed=false`.

## 9. Required E20 outputs after future Human-authorized execution

- per-seed/per-fold Stage-A and Stage-B registry records;
- selected and refit epochs;
- checkpoint and config checksums;
- per-seed/fold predictions and metrics;
- pooled metrics for each configuration and seed;
- across-seed mean and SD;
- paired finalist-minus-control deltas for every seed/fold;
- worst-fold and fold-consistency evidence;
- provenance, leakage, population, scaler, failure-ledger, and Test-firewall signoff;
- execution manifest and final Human-review artifact.

## 10. Selection and promotion policy

- Never choose the best seed.
- Never retune from seed behavior.
- Report all seeds individually and together.
- Use paired seed evidence; no weighted composite score.
- Preserve the locked Wave 2 RMSE, MAE, worst-fold, and fold-consistency guardrails.
- Provenance, checksum, population equality, ordered target IDs, fold-local scaling, leakage, and Test firewall must PASS.
- Final promotion requires explicit Human review.

## 11. Hard stops

Stop on any seed-42 reuse mismatch, missing source evidence, duplicate canonical run, configuration drift, unequal target population, scaler leakage, optimizer/scheduler state reuse, Test access, unapproved E19 dependency, or training without separate Human authorization.

```text
E20_IMPLEMENTATION_AUTHORIZED = NO
E20_TRAINING_AUTHORIZED = NO
TEST_ACCESS = NO
```
