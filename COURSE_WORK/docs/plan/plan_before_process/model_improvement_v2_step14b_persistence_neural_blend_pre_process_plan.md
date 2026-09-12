# MODEL_IMPROVEMENT-v2 Step 14-B Persistence-Neural Blend Pre-Process Plan

## 1. Authority and locked entry state

This plan follows:

- `docs/Plan_improve_model.md`, especially Phase I15.2 and Step 14;
- `docs/plan/plan_before_process/model_improvement_v2_pre_process_plan.md`;
- `docs/plan/plan_before_process/model_improvement_v2_wave2_pre_process_plan.md`;
- `docs/plan/plan_before_process/model_improvement_v2_step14a_seed_ensemble_pre_process_plan.md`;
- the Human decision `STEP14A_HUMAN_DECISION = SELECT_FINALIST_ENSEMBLE`.

The accepted neural source for this experiment is locked as:

```text
STEP14B_NEURAL_SOURCE = STEP14A_FINALIST_ENSEMBLE
member_candidate = TR_C2_ALT_LOOKBACK_E14_M1
member_seeds = [42, 123, 2026]
member_weights = [1/3, 1/3, 1/3]
pooled_rmse_wh = 58.57180673355406
pooled_mae_wh = 25.545269885543682
pooled_r2 = 0.5967752914420708
worst_fold_rmse_wh = 67.53675009939884
fold_rmse_sd_wh_ddof1 = 8.879494624591965
```

```text
STEP14B_PLAN_STATUS = PREPARED_FOR_HUMAN_REVIEW
STEP14B_EXPECTED_NEW_RUNS = 0
STEP14B_IMPLEMENTATION_AUTHORIZED = NO
STEP14B_EVALUATION_AUTHORIZED = NO
STEP14B_TRAINING_AUTHORIZED = NO
TEST_STATUS = NOT_ACCESSED
```

This document does not calculate blend predictions or metrics and does not authorize evaluation.

## 2. Scientific question and primary change

Scientific question: can a fixed, pre-registered convex blend with the causal Persistence baseline improve the Human-selected three-seed neural ensemble on rolling-origin development evidence without sacrificing its MAE, worst-fold, or fold-consistency guardrails?

The only primary change is the fixed persistence-neural mixing coefficient `alpha`:

```text
prediction_blend = alpha * FINALIST_ENSEMBLE
                 + (1 - alpha) * PERSISTENCE

alpha_matrix = [0.25, 0.50, 0.75]
```

The matrix is closed. No additional alpha, continuous search, interpolation, adaptive gate, per-fold weight, per-seed weight, per-target weight, calibration, clipping, or stacking is allowed. In particular, the Test-diagnostic 0.6-0.7 region must not be evaluated or used.

The neural-only source is the read-only control. The three fixed alpha values are the only challengers:

| Role | Candidate ID | Alpha neural | Weight persistence |
|---|---|---:|---:|
| CONTROL | `STEP14A_FINALIST_ENSEMBLE` | 1.00 | 0.00 |
| CHALLENGER | `STEP14B_BLEND_ALPHA_025` | 0.25 | 0.75 |
| CHALLENGER | `STEP14B_BLEND_ALPHA_050` | 0.50 | 0.50 |
| CHALLENGER | `STEP14B_BLEND_ALPHA_075` | 0.75 | 0.25 |

## 3. Locked neural source

The neural prediction for each target is reconstructed deterministically from the nine immutable E14-M1 development prediction files already locked by Step 14-A. SHA-256 is over the exact CSV bytes.

| Seed | Fold | Rows | Prediction source | SHA-256 |
|---:|---|---:|---|---|
| 42 | RO1 | 987 | `artifacts/model_improvement_v2/experiments/E14/predictions/outer_predictions_TR_C2_ALT_LOOKBACK_E14_M1_RO1.csv` | `171f61551033e3ba288459d8c5f2a85f73f3c1fc09db75b39501d267c94f68ae` |
| 42 | RO2 | 987 | `artifacts/model_improvement_v2/experiments/E14/predictions/outer_predictions_TR_C2_ALT_LOOKBACK_E14_M1_RO2.csv` | `090bd85b1553eb1f71237659e768d8b137374d4c39f0337967494f3849ddd519` |
| 42 | RO3 | 986 | `artifacts/model_improvement_v2/experiments/E14/predictions/outer_predictions_TR_C2_ALT_LOOKBACK_E14_M1_RO3.csv` | `c1d13c571f097955c194123bcd8cb106f6510687a33645838fd6800d336220d4` |
| 123 | RO1 | 987 | `artifacts/model_improvement_v2/experiments/E20/seed_123/predictions/outer_predictions_TR_C2_ALT_LOOKBACK_E14_M1_RO1.csv` | `7aa5e3d61f305f0adb4038baf840049a82638a0f4d5ebaeef79942c2504abc41` |
| 123 | RO2 | 987 | `artifacts/model_improvement_v2/experiments/E20/seed_123/predictions/outer_predictions_TR_C2_ALT_LOOKBACK_E14_M1_RO2.csv` | `db18d89c1dbdb4810f2088b205f579c3b8d30145c8b73c3da93503af58e471d1` |
| 123 | RO3 | 986 | `artifacts/model_improvement_v2/experiments/E20/seed_123/predictions/outer_predictions_TR_C2_ALT_LOOKBACK_E14_M1_RO3.csv` | `74789275c5609d827685b45f30dcc8952814177d6a1ea6476d597b58cd69700b` |
| 2026 | RO1 | 987 | `artifacts/model_improvement_v2/experiments/E20/seed_2026/predictions/outer_predictions_TR_C2_ALT_LOOKBACK_E14_M1_RO1.csv` | `9ec8205713cab0dbd8f92e3da1d24a29738d93bc33f67ca155fdef6ee752a6d8` |
| 2026 | RO2 | 987 | `artifacts/model_improvement_v2/experiments/E20/seed_2026/predictions/outer_predictions_TR_C2_ALT_LOOKBACK_E14_M1_RO2.csv` | `2739441306e69123a413f763cfd936c29466ec9130d510e0a852fbb01ef30be5` |
| 2026 | RO3 | 986 | `artifacts/model_improvement_v2/experiments/E20/seed_2026/predictions/outer_predictions_TR_C2_ALT_LOOKBACK_E14_M1_RO3.csv` | `10ec18fec040369d2ab16708e3c89e8eda202145fca9acfa2f3680e79b704f46` |

The Step 14-A plan containing the complete source and population lock has SHA-256:

```text
e2a8762c8b463674f59c303de52f12300128083d11002852ccb4ca8b6e703fad
```

At evaluation time, the neural ensemble must be recomputed as the exact raw-Wh arithmetic mean of all three seeds and must reproduce the Human-accepted Step 14-A metrics before any blend is evaluated. A mismatch is a hard stop, not permission to modify Step 14-A evidence.

## 4. Locked Persistence source

The canonical Persistence source is the E01 development prediction bundle:

| Fold | Rows | Prediction source | SHA-256 |
|---|---:|---|---|
| RO1 | 987 | `artifacts/model_improvement_v2/experiments/E01/predictions/outer_predictions_PERSISTENCE_LAST_VALUE_RO1.csv` | `42aa1026148c7c5facbcb53484316af303bf6b4d67113a1e9beda5316f3e4033` |
| RO2 | 987 | `artifacts/model_improvement_v2/experiments/E01/predictions/outer_predictions_PERSISTENCE_LAST_VALUE_RO2.csv` | `91e31b9c2deae3becfcb36d1709f81b643e4adae25b1a68f6cd24ea15028faf2` |
| RO3 | 986 | `artifacts/model_improvement_v2/experiments/E01/predictions/outer_predictions_PERSISTENCE_LAST_VALUE_RO3.csv` | `10889c1b6e1a34f2790fbbeb47c9e04afd2380bd144221b9ea53d952e04ad7ff` |

Preparation verified that the corresponding E20 seed-123 and seed-2026 Persistence files are byte-identical to these sources for every fold. Persistence is deterministic and is included exactly once per target; it is not averaged three times or treated as a seed-dependent model.

## 5. Ordered population and alignment lock

The neural ensemble and Persistence must have identical ordered `target_id` and `y_true_wh` arrays before blending.

| Fold | Count | Ordered target-ID SHA-256 | Ordered `y_true_wh` SHA-256 | Status |
|---|---:|---|---|---|
| RO1 | 987 | `5fd1662d17cd200f2ab6f46cf8ae61bfe3369fbab30f0840a0dd95ad139e2bd3` | `11225956b31d0e0b288747c95525364c20bded506f46b0ff761a16141b3ca468` | PASS |
| RO2 | 987 | `0752a3ebc071222126780e51e1cf7a1768495acbb503b019b3062f2f98228d16` | `c367acf778d44c454f4c2fcf79fc71d648f86cac27001b94aef07134713a87f8` | PASS |
| RO3 | 986 | `9776be7207a88e76a708648481bebb2f797d851f982e40178bbd7daacbe81a1a` | `7ba6001a6280e0e6f4c9519b6c3772e380d06f2afb281e2b5fd16d9053fb2b3b` | PASS |

The hashes use the Step 14-A definition: SHA-256 over newline-joined ordered CSV string values without a trailing newline. They do not replace upstream canonical fingerprints.

Evaluation must use an equality assertion, not a permissive join. No target may be dropped, duplicated, reordered, imputed, or backfilled. Each candidate is evaluated on the identical 2,960-target development population.

## 6. Development-only evaluation contract

Only after separate Human evaluation approval may implementation:

1. verify all locked neural and Persistence source checksums;
2. validate candidate, fold, schema, target order, truth values, and row counts;
3. reconstruct the accepted neural ensemble and reproduce its locked metrics;
4. calculate the three blend predictions in raw Wh using only the locked alpha matrix;
5. calculate canonical metrics on RO1, RO2, RO3 and the concatenated 2,960-target population;
6. write only the isolated Step 14-B artifacts in Section 8;
7. stop at a Human decision gate.

No checkpoint/model/scaler may be loaded. No training, refit, model forward, new inference, scaler fitting, or prediction regeneration is allowed.

## 7. Metrics, ranking, and promotion gates

Required for the neural control and every alpha challenger:

- pooled RMSE, MAE, and R2;
- RO1, RO2, and RO3 RMSE;
- worst-fold RMSE;
- fold-RMSE sample SD (`ddof=1`), matching the approved Step 14-A contract.

Every challenger is compared directly with the accepted neural ensemble. Lower pooled RMSE ranks eligible challengers, but no result is automatically promoted and no weighted composite score is permitted.

A blend is eligible for Human promotion only if every gate passes:

```text
neural_pooled_rmse_wh - blend_pooled_rmse_wh >= 0.10 Wh
blend_pooled_mae_wh - neural_pooled_mae_wh <= 0.25 Wh
blend_worst_fold_rmse_wh - neural_worst_fold_rmse_wh <= 0.50 Wh
blend_fold_rmse_sd_wh - neural_fold_rmse_sd_wh <= 0.50 Wh
source provenance/checksum = PASS
ordered population equality = PASS
metric contract = PASS
Test firewall = PASS
```

If multiple fixed-alpha candidates are eligible, report all of them and recommend the eligible candidate with the lowest pooled RMSE for Human review. If none is eligible, recommend retaining `STEP14A_FINALIST_ENSEMBLE`. Alpha selection must not use Test evidence or any metric outside this locked contract.

## 8. Isolated namespace and future artifacts

Reserved namespace:

```text
COURSE_WORK/artifacts/model_improvement_v2/experiments/STEP14B/
```

Only separately authorized implementation/evaluation may create:

```text
step14b_config_snapshot.json
step14b_prediction_source_ledger.json
step14b_population_audit.json
predictions/step14b_blend_alpha_025_RO1.csv
predictions/step14b_blend_alpha_025_RO2.csv
predictions/step14b_blend_alpha_025_RO3.csv
predictions/step14b_blend_alpha_050_RO1.csv
predictions/step14b_blend_alpha_050_RO2.csv
predictions/step14b_blend_alpha_050_RO3.csv
predictions/step14b_blend_alpha_075_RO1.csv
predictions/step14b_blend_alpha_075_RO2.csv
predictions/step14b_blend_alpha_075_RO3.csv
step14b_fold_metrics.csv
step14b_pooled_metrics.csv
step14b_blend_comparison.json
step14b_execution_manifest.json
step14b_signoff.json
step14b_human_decision.json
```

Expected new training runs, registry records, checkpoints, optimizer states, scheduler states, and training histories are all zero. E01, E14, E20, and Step 14-A evidence remain immutable.

## 9. Test firewall

Allowed reads are restricted to the nine neural-member files, three canonical Persistence files, the Step 14-A plan/source lock, and frozen development-only provenance needed to validate them.

Forbidden:

- any path under `artifacts/final_test/` or any Phase 47 Test output;
- Test rows, target IDs, truth, predictions, metrics, scalers, thresholds, or weights;
- old-Test diagnostic blend values as evidence;
- any alpha outside `[0.25, 0.50, 0.75]`, including post-hoc 0.6-0.7 values;
- selecting a different neural member, seed subset, Persistence variant, or per-fold alpha after seeing results.

Any Test path resolution or unlisted source read is a hard stop.

## 10. Hard stops and Human gates

Stop without evaluation if:

- a source checksum, schema, candidate, fold, count, target order, or truth vector mismatches;
- the neural ensemble fails to reproduce the accepted Step 14-A metrics;
- Persistence differs from its locked E01 source;
- an alpha is added, removed, tuned, or applied inconsistently;
- the source population is changed;
- a model, checkpoint, or scaler would be loaded;
- any upstream artifact would be modified;
- Test is accessed.

```text
STEP14B_IMPLEMENTATION_AUTHORIZED = NO
STEP14B_EVALUATION_AUTHORIZED = NO
STEP14B_TRAINING_AUTHORIZED = NO
STEP14B_EXPECTED_NEW_RUNS = 0
TEST_ACCESS = NO
```

## 11. Definition of ready for implementation/evaluation

Human review must explicitly accept:

- the Step 14-A neural source and locked metrics;
- the three canonical Persistence files and checksums;
- the closed alpha matrix `[0.25, 0.50, 0.75]`;
- the identical ordered 2,960-target population;
- the metric and promotion contract;
- the isolated `STEP14B` namespace;
- zero training, zero new model inference, and the Test firewall.

Implementation must provide focused checksum, alignment, arithmetic, alpha-matrix, metric, namespace, and Test-firewall tests plus a dry preflight. Evaluation remains separately gated.
