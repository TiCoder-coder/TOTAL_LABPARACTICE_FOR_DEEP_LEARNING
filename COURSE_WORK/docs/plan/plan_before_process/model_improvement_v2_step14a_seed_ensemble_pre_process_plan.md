# MODEL_IMPROVEMENT-v2 Step 14-A Seed Ensemble Pre-Process Plan

## 1. Authority and current state

This plan follows:

- `docs/Plan_improve_model.md`, especially Phase I15 and Step 14;
- `docs/plan/plan_before_process/model_improvement_v2_pre_process_plan.md`;
- `docs/plan/plan_before_process/model_improvement_v2_wave2_pre_process_plan.md`;
- `docs/plan/plan_before_process/model_improvement_v2_e20_pre_process_plan.md`;
- the Human decision `E20_HUMAN_DECISION = REJECT_FINALIST_CONFIRMATION`.

```text
STEP14A_PLAN_STATUS = PREPARED_FOR_HUMAN_REVIEW
E20_FINALIST_CONFIRMATION = REJECTED
STEP14A_EXPECTED_NEW_RUNS = 0
STEP14A_EVALUATION_AUTHORIZED = NO
STEP14A_TRAINING_AUTHORIZED = NO
TEST_STATUS = NOT_ACCESSED
```

The E20 decision prevents `TR_C2_ALT_LOOKBACK_E14_M1` from being treated as a confirmed multi-seed winner. It does not invalidate its completed development predictions. Step 14-A may use those immutable predictions only as a fixed finalist-ensemble candidate against the matched E01-equivalent control ensemble.

This plan does not authorize calculation of ensemble predictions or metrics. It does not open the persistence-neural blend.

## 2. Scientific question and primary change

Scientific question: does deterministic equal-weight averaging across all three locked seeds reduce seed-specific variance enough for the E14-M1 ensemble to outperform the matched E01-equivalent ensemble under the existing Wave 2 promotion guardrails?

The only new policy is prediction aggregation across seeds:

```text
prediction_ensemble(target) =
    (prediction_seed42(target)
     + prediction_seed123(target)
     + prediction_seed2026(target)) / 3

weights = [1/3, 1/3, 1/3]
seeds = [42, 123, 2026]
```

No seed may be selected, omitted, duplicated, reweighted, or replaced. In particular, seed 2026 must remain in both ensembles. No weight tuning, stacking, calibration, clipping, or persistence blending is permitted.

## 3. Locked ensemble identities

### 3.1 Control ensemble

```text
ensemble_id = STEP14A_CONTROL_ENSEMBLE
member_candidate = TR_C2_ALT_LOOKBACK
member_configuration = E01-equivalent
member_seeds = [42, 123, 2026]
weights = [1/3, 1/3, 1/3]
```

### 3.2 Finalist ensemble

```text
ensemble_id = STEP14A_FINALIST_ENSEMBLE
member_candidate = TR_C2_ALT_LOOKBACK_E14_M1
member_configuration = E14-M1 locked configuration
member_seeds = [42, 123, 2026]
weights = [1/3, 1/3, 1/3]
```

Both sides use the same folds, seed set, aggregation formula, ordered target population, and metric implementation. E14-M1 remains a candidate for this ensemble comparison and is not silently reinstated as the single-model winner.

## 4. Immutable prediction-source ledger

All paths are relative to `COURSE_WORK/`. SHA-256 is over the exact source CSV bytes. Every source is read-only.

| Role | Seed | Fold | Rows | Prediction source | SHA-256 |
|---|---:|---|---:|---|---|
| CONTROL | 42 | RO1 | 987 | `artifacts/model_improvement_v2/experiments/E01/predictions/outer_predictions_TR_C2_ALT_LOOKBACK_RO1.csv` | `a877ef5f9496e14d9a51306b7c7a23bd17bd3436609fe90592121582c11f0196` |
| CONTROL | 42 | RO2 | 987 | `artifacts/model_improvement_v2/experiments/E01/predictions/outer_predictions_TR_C2_ALT_LOOKBACK_RO2.csv` | `f6cba6d6d3243ffbccb8c1ce3f19600d074e4f50afb10ad9d10c7492381567e1` |
| CONTROL | 42 | RO3 | 986 | `artifacts/model_improvement_v2/experiments/E01/predictions/outer_predictions_TR_C2_ALT_LOOKBACK_RO3.csv` | `0183f23124ddd2814a931e12c02dbbf114fd0914f1bf0b07bf910d8a7e7ba4b0` |
| CONTROL | 123 | RO1 | 987 | `artifacts/model_improvement_v2/experiments/E20/seed_123/predictions/outer_predictions_TR_C2_ALT_LOOKBACK_RO1.csv` | `99df73ee76d787933ad67f33651c36b03a908a0033b4ddff3ca7fc218e890fed` |
| CONTROL | 123 | RO2 | 987 | `artifacts/model_improvement_v2/experiments/E20/seed_123/predictions/outer_predictions_TR_C2_ALT_LOOKBACK_RO2.csv` | `704a7dd75722839d31aa19a065a4455705ae471e8700427495fe67497eb75a83` |
| CONTROL | 123 | RO3 | 986 | `artifacts/model_improvement_v2/experiments/E20/seed_123/predictions/outer_predictions_TR_C2_ALT_LOOKBACK_RO3.csv` | `7a9c98a3de6fde57369bad86f7587f260f7e2b387b810d5d20029bbf903a46a9` |
| CONTROL | 2026 | RO1 | 987 | `artifacts/model_improvement_v2/experiments/E20/seed_2026/predictions/outer_predictions_TR_C2_ALT_LOOKBACK_RO1.csv` | `dbbebcd9f3b57794182c6341fe6a809b22b0a35b6cd9be27780c6044b62ccf9a` |
| CONTROL | 2026 | RO2 | 987 | `artifacts/model_improvement_v2/experiments/E20/seed_2026/predictions/outer_predictions_TR_C2_ALT_LOOKBACK_RO2.csv` | `8fc0e86f782a1501a45b21e384fba822bc85ec5c800a27b1f092131706a145b1` |
| CONTROL | 2026 | RO3 | 986 | `artifacts/model_improvement_v2/experiments/E20/seed_2026/predictions/outer_predictions_TR_C2_ALT_LOOKBACK_RO3.csv` | `b0973488cef6885a96b71a11180ac0d7826954ce221fc296e00f33af0f826189` |
| FINALIST | 42 | RO1 | 987 | `artifacts/model_improvement_v2/experiments/E14/predictions/outer_predictions_TR_C2_ALT_LOOKBACK_E14_M1_RO1.csv` | `171f61551033e3ba288459d8c5f2a85f73f3c1fc09db75b39501d267c94f68ae` |
| FINALIST | 42 | RO2 | 987 | `artifacts/model_improvement_v2/experiments/E14/predictions/outer_predictions_TR_C2_ALT_LOOKBACK_E14_M1_RO2.csv` | `090bd85b1553eb1f71237659e768d8b137374d4c39f0337967494f3849ddd519` |
| FINALIST | 42 | RO3 | 986 | `artifacts/model_improvement_v2/experiments/E14/predictions/outer_predictions_TR_C2_ALT_LOOKBACK_E14_M1_RO3.csv` | `c1d13c571f097955c194123bcd8cb106f6510687a33645838fd6800d336220d4` |
| FINALIST | 123 | RO1 | 987 | `artifacts/model_improvement_v2/experiments/E20/seed_123/predictions/outer_predictions_TR_C2_ALT_LOOKBACK_E14_M1_RO1.csv` | `7aa5e3d61f305f0adb4038baf840049a82638a0f4d5ebaeef79942c2504abc41` |
| FINALIST | 123 | RO2 | 987 | `artifacts/model_improvement_v2/experiments/E20/seed_123/predictions/outer_predictions_TR_C2_ALT_LOOKBACK_E14_M1_RO2.csv` | `db18d89c1dbdb4810f2088b205f579c3b8d30145c8b73c3da93503af58e471d1` |
| FINALIST | 123 | RO3 | 986 | `artifacts/model_improvement_v2/experiments/E20/seed_123/predictions/outer_predictions_TR_C2_ALT_LOOKBACK_E14_M1_RO3.csv` | `74789275c5609d827685b45f30dcc8952814177d6a1ea6476d597b58cd69700b` |
| FINALIST | 2026 | RO1 | 987 | `artifacts/model_improvement_v2/experiments/E20/seed_2026/predictions/outer_predictions_TR_C2_ALT_LOOKBACK_E14_M1_RO1.csv` | `9ec8205713cab0dbd8f92e3da1d24a29738d93bc33f67ca155fdef6ee752a6d8` |
| FINALIST | 2026 | RO2 | 987 | `artifacts/model_improvement_v2/experiments/E20/seed_2026/predictions/outer_predictions_TR_C2_ALT_LOOKBACK_E14_M1_RO2.csv` | `2739441306e69123a413f763cfd936c29466ec9130d510e0a852fbb01ef30be5` |
| FINALIST | 2026 | RO3 | 986 | `artifacts/model_improvement_v2/experiments/E20/seed_2026/predictions/outer_predictions_TR_C2_ALT_LOOKBACK_E14_M1_RO3.csv` | `10ec18fec040369d2ab16708e3c89e8eda202145fca9acfa2f3680e79b704f46` |

Evaluation must fail before producing any derived artifact if any source path, byte checksum, candidate ID, seed/fold identity, column schema, row count, target order, or truth vector differs from this lock.

## 5. Ordered population lock

The preparation audit established exact ordered equality of `target_id` and `y_true_wh` across all six member streams in each fold.

| Fold | Count | Ordered target-ID SHA-256 | Ordered `y_true_wh` SHA-256 | Status |
|---|---:|---|---|---|
| RO1 | 987 | `5fd1662d17cd200f2ab6f46cf8ae61bfe3369fbab30f0840a0dd95ad139e2bd3` | `11225956b31d0e0b288747c95525364c20bded506f46b0ff761a16141b3ca468` | PASS |
| RO2 | 987 | `0752a3ebc071222126780e51e1cf7a1768495acbb503b019b3062f2f98228d16` | `c367acf778d44c454f4c2fcf79fc71d648f86cac27001b94aef07134713a87f8` | PASS |
| RO3 | 986 | `9776be7207a88e76a708648481bebb2f797d851f982e40178bbd7daacbe81a1a` | `7ba6001a6280e0e6f4c9519b6c3772e380d06f2afb281e2b5fd16d9053fb2b3b` | PASS |

The hashes above are explicitly defined as SHA-256 over newline-joined ordered CSV string values, without a trailing newline. They are Step 14-A audit hashes and do not replace upstream canonical population fingerprints.

Ensemble averaging must be an inner join with an equality assertion, never a permissive join. No row may be dropped, imputed, reordered, or duplicated. Pooled evaluation concatenates RO1, RO2, and RO3 exactly once for a total of 2,960 targets per ensemble.

## 6. Development-only evaluation contract

After separate Human authorization, deterministic evaluation may only:

1. verify the 18 source files and locked checksums;
2. validate exact per-fold candidate/seed identity, ordered target IDs, and `y_true_wh` equality;
3. compute each ensemble prediction in raw Wh using the fixed arithmetic mean;
4. calculate per-fold and pooled development metrics using the existing canonical metric definitions;
5. write derived Step 14-A artifacts only in the isolated namespace in Section 8;
6. stop at a Human promotion decision gate.

No model or checkpoint may be loaded. No model forward pass, training, refit, scaler fitting, prediction regeneration, or Test access is part of Step 14-A.

## 7. Metrics and promotion guardrails

Required metrics for both ensembles:

- pooled RMSE, MAE, and R2 over the 2,960 ordered development targets;
- RO1, RO2, and RO3 RMSE;
- worst-fold RMSE = maximum fold RMSE;
- fold consistency = sample SD (`ddof=1`) of the three fold RMSE values.

Report finalist-minus-control deltas. Lower is better for RMSE, MAE, worst-fold RMSE, and fold-RMSE SD. Do not create a weighted composite score.

The finalist ensemble is eligible for Human promotion only if all gates pass:

```text
control_pooled_rmse_wh - finalist_pooled_rmse_wh >= 0.10 Wh
finalist_pooled_mae_wh - control_pooled_mae_wh <= 0.25 Wh
finalist_worst_fold_rmse_wh - control_worst_fold_rmse_wh <= 0.50 Wh
finalist_fold_rmse_sd_wh - control_fold_rmse_sd_wh <= 0.50 Wh
source provenance/checksum = PASS
ordered population equality = PASS
metric contract = PASS
Test firewall = PASS
```

Passing metrics do not automatically promote an ensemble. Final promotion remains a Human decision. Failure of any gate requires a reject recommendation and cannot be repaired by changing seeds or weights.

## 8. Isolated namespace and future artifacts

Reserved namespace:

```text
COURSE_WORK/artifacts/model_improvement_v2/experiments/STEP14A/
```

Only a separately authorized implementation/evaluation may create:

```text
step14a_config_snapshot.json
step14a_prediction_source_ledger.json
step14a_population_audit.json
predictions/step14a_control_seed_ensemble_RO1.csv
predictions/step14a_control_seed_ensemble_RO2.csv
predictions/step14a_control_seed_ensemble_RO3.csv
predictions/step14a_finalist_seed_ensemble_RO1.csv
predictions/step14a_finalist_seed_ensemble_RO2.csv
predictions/step14a_finalist_seed_ensemble_RO3.csv
step14a_fold_metrics.csv
step14a_pooled_metrics.csv
step14a_ensemble_comparison.json
step14a_execution_manifest.json
step14a_signoff.json
step14a_human_decision.json
```

No training registry entry, run ID, checkpoint, optimizer state, scheduler state, or training history is expected. E01, E14, and E20 artifacts remain immutable.

## 9. Test firewall

Allowed reads are limited to the 18 development prediction CSVs in Section 4 and the frozen development provenance/configuration artifacts needed to validate them.

Forbidden:

- every file under `artifacts/final_test/` or any Phase 47 Test output;
- Test rows, targets, predictions, scalers, metrics, thresholds, or metadata used for ranking;
- old-Test diagnostic ensemble values as promotion evidence;
- Test-derived weights or post-hoc weights, including the previously observed 0.6-0.7 range;
- best-seed selection or exclusion of seed 2026;
- persistence-neural blending in Step 14-A.

Any attempted Test path resolution, Test-labeled target, source mismatch, or access to an unlisted prediction file is a hard stop.

## 10. Hard stops and Human gates

Stop without evaluation if:

- any of the 18 source checksums fails;
- any fold has unequal ordered target IDs or truth values;
- a seed is missing, excluded, duplicated, or assigned a non-equal weight;
- an input is regenerated instead of read from the frozen source ledger;
- any metric or promotion threshold is changed after ensemble results are seen;
- persistence blending is introduced;
- Test is accessed;
- any E01, E14, or E20 scientific artifact would be modified.

```text
STEP14A_IMPLEMENTATION_AUTHORIZED = NO
STEP14A_EVALUATION_AUTHORIZED = NO
STEP14A_TRAINING_AUTHORIZED = NO
STEP14A_EXPECTED_NEW_RUNS = 0
TEST_ACCESS = NO
```

## 11. Definition of ready for evaluation implementation

Step 14-A implementation may begin only after Human review explicitly accepts:

- this immutable 18-file prediction-source ledger;
- equal weights `[1/3, 1/3, 1/3]` and all three seeds;
- the population and metric contracts;
- the numeric promotion guardrails;
- the isolated `STEP14A` namespace;
- zero training, zero new inference, and the Test firewall.

Implementation must provide focused source-integrity, population-equality, arithmetic, metric, namespace, and Test-firewall tests plus a dry preflight. Evaluation remains separately gated and must not be implied by approval to implement.
