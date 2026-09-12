# MODEL_IMPROVEMENT-v2 Fine-Tuning Pre-Process Plan

## 1. Status and authority

This plan follows `docs/Plan_improve_model.md`, the authoritative Wave 2 pre-process plan, and the Human-accepted E18 eligibility decision.

```text
E18_FINETUNING_ELIGIBILITY = ACCEPTED
ACTIVE_INCUMBENT = TR_C2_ALT_LOOKBACK_E14_M1
EXPECTED_NEW_RUNS = 0
FINETUNING_TRAINING_AUTHORIZED = NO
TEST_STATUS = NOT_ACCESSED
```

This document does not authorize FT-A, FT-B, FT-C, training, inference, Test access, checkpoint creation, or automatic opening of E19.

## 2. Scientific question

Can fold-matched E01 Stage-B initialization improve rolling-origin development performance or convergence versus the exact scratch-trained E14-M1 comparator when every non-initialization scientific field is held fixed?

Any future executable experiment must declare initialization/warm-start policy as its only primary change.

## 3. Verified source checkpoints

Only the following fold-matched E01 Stage-B checkpoints are eligible sources:

| Fold | Run ID | Epoch | SHA-256 |
|---|---|---:|---|
| RO1 | `RUN_V2_TR_E01_RO1_B_0004_533B5F33` | 2 | `6aa201d63388c1922c5190c1869b640e945dc2a95c6a03a69f3307912ba478a3` |
| RO2 | `RUN_V2_TR_E01_RO2_B_0005_19A72D17` | 8 | `47a8588f84ad8b1143c317dafe76830c1bfa56aa0ff1e70227b7bbd070df6140` |
| RO3 | `RUN_V2_TR_E01_RO3_B_0006_AA5D66A6` | 21 | `3f85415a4dc157a552b19bab6c17334420d307cb5b398139a14e5c293312981b` |

Cross-fold mapping is forbidden. Phase47/Test checkpoints are forbidden.

## 4. Exact scratch comparator

The comparator is the fold-matched scratch-trained `TR_C2_ALT_LOOKBACK_E14_M1` evidence accepted by Human decision:

- `d_model=64`, `num_heads=4`, `ffn_dim=256`, `num_layers=2`, Post-LN;
- DIRECT linear head, `FS2_TF1`, 33 features, lookback 72, horizon 1, WB0;
- AdamW, LR `2e-4`, batch 16, weight decay `1e-3`, dropout `0.10`;
- scheduler OFF, hybrid level-plus-delta loss with `lambda_delta=0.10`;
- RO1/RO2/RO3, fold-local X/Y scaling, seed 42, Test not accessed.

The existing E14-M1 run/checkpoint evidence is read-only. A future plan must prove exact population and budget comparability before reuse.

## 5. Reuse scope

For the current incumbent, E01 and E14-M1 have identical model state-dict keys/shapes, architecture, feature order/count, input projection, lookback, horizon, boundary protocol, folds, and population. Therefore:

```text
CURRENT_TECHNICAL_REUSE_SCOPE = FULL_MODEL
CHECKPOINT_REUSE_CLASSIFICATION = EXACT_REUSE_ALLOWED
OPTIMIZER_STATE_REUSE = FORBIDDEN
SCHEDULER_STATE_REUSE = FORBIDDEN
```

If a future Human decision introduces a new residual/MLP/gate head, full-model reuse no longer applies to that head. Only the verified encoder plus input projection may be reused, the new head must be freshly initialized, and an exact scratch comparator for that new architecture must be established.

## 6. FT-A / FT-B / FT-C validity gate

The staged FT-A/B/C sequence in the source plan is designed for adaptation after a new residual/MLP/gate head is introduced. E03, E04, and E05 were rejected, so no such head is active.

```text
FT_A_VALIDITY = DEFERRED
FT_B_VALIDITY = DEFERRED
FT_C_VALIDITY = DEFERRED
REASON = NO_HUMAN_PROMOTED_NEW_HEAD
```

Consequently, the source-plan FT-A head-only adaptation has no current target head, and FT-B/FT-C must not open. The source-plan learning rates, scheduler choices, and epoch ranges are not executable locks.

A same-head full-model warm-start study would be a different initialization ablation. It requires a separate Human-approved experiment specification before implementation.

## 7. Requirements for any future separate FT plan

Before implementation, Human review must lock:

1. reuse scope: full model, or encoder plus input projection with a new head;
2. exactly one primary scientific change;
3. fold-matched source and scratch-control mapping;
4. frozen and trainable parameter groups for every stage;
5. fresh optimizer and scheduler state for every independent run/stage;
6. all learning rates, scheduler parameters, warmup, epochs, stopping rules, and transition gates;
7. identical RO folds, ordered target IDs, feature order, population, fold-local scalers, seed/data order, loss, and evaluation protocol;
8. failure/recovery/no-retrain lifecycle and isolated V2 artifact namespace.

No values may be selected using Test.

## 8. Promotion and stop gates

Any future eligible challenger must satisfy the locked Wave 2 gates against its exact scratch comparator:

- pooled RMSE improvement at least `0.10 Wh`;
- pooled MAE degradation no more than `0.25 Wh`;
- worst-fold RMSE degradation no more than `0.50 Wh`;
- fold-RMSE SD degradation no more than `0.50 Wh`;
- provenance, checksum, population equality, ordered target IDs, feature compatibility, fold-local scaling, leakage, and Test firewall all PASS;
- explicit Human promotion.

Failure of FT-A closes FT-B and FT-C. Failure of FT-B closes FT-C. Convergence speed alone cannot promote a candidate.

## 9. Current recommendation

```text
FT_PLAN_STATUS = COMPLETE_NON_EXECUTABLE
FT_RECOMMENDATION = DEFER
FT_TRAINING_AUTHORIZED = NO
HUMAN_DECISION_REQUIRED = YES
```

Resume planning only after Human explicitly approves either a new-head staged experiment or a same-head full-model initialization ablation.
