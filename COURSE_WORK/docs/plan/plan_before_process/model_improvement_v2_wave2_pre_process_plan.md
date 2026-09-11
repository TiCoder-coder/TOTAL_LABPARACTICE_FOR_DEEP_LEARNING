# MODEL_IMPROVEMENT-v2 Wave 2 Pre-Process Plan

## 1. Purpose and authority

This document defines the planning and review contract for Wave 2 after completion of E00–E09. It is derived from:

- `COURSE_WORK/docs/Plan_improve_model.md`;
- `COURSE_WORK/docs/plan/plan_before_process/model_improvement_v2_pre_process_plan.md`;
- accepted V2 development evidence from E01–E09.

This document does not modify either source plan and does not authorize implementation, training, inference, checkpoint loading, Test access, artifact regeneration, commit, or push.

The Human has authorized Wave 2 planning only. Every experiment still requires a separate implementation-preparation task and explicit Human training approval.

## 2. Wave 1 closure and accepted baseline

Wave 1 is closed. E02–E09 were not promoted. Rejected feature, prediction, head, gate, LR, scheduler, SGD, and SGDM choices must not silently become Wave 2 defaults.

The global accepted development baseline remains:

```yaml
source_experiment: E01
candidate_id: TR_C2_ALT_LOOKBACK
prediction_formulation: DIRECT
feature_set: FS2_TF1
feature_count: 33
lookback: 72
horizon: 1
boundary_protocol: WB0_CONTEXT_CARRY_OVER
rolling_origin_protocol: RO3_EXPANDING_PRETEST-v1
folds: [RO1, RO2, RO3]
fold_local_x_scaling: true
fold_local_y_scaling: true
optimizer: AdamW
learning_rate: 3e-4
weight_decay: 1e-3
scheduler: OFF
loss: MSE
seed: 42
pooled_rmse_wh: 59.85291570400546
pooled_mae_wh: 26.650501720144604
pooled_r2: 0.5789433617557903
test_status: NOT_ACCESSED
```

E01 remains the global baseline until a later candidate is explicitly promoted by Human review. Within an experiment, a population-matched local control may need to be retrained when a causal feature changes eligibility. Such a local control does not replace E01 globally by itself.

The Wave 1 promotion tolerance of `0.10 Wh` and threshold `59.75291570400546 Wh` must be explicitly reaffirmed for Wave 2 before training. Until then:

```text
WAVE2_PROMOTION_RMSE_TOLERANCE = HUMAN_TO_LOCK_BEFORE_TRAINING
WAVE2_POOLED_MAE_TOLERANCE = HUMAN_TO_LOCK_BEFORE_TRAINING
WAVE2_WORST_FOLD_TOLERANCE = HUMAN_TO_LOCK_BEFORE_TRAINING
WAVE2_FOLD_CONSISTENCY_RULE = HUMAN_TO_LOCK_BEFORE_TRAINING
```

## 3. Non-negotiable scientific firewall

Allowed evidence is limited to Train and Validation chronology under the existing RO1/RO2/RO3 expanding pre-Test protocol.

Forbidden:

- loading any Test row, feature, target, scaler, prediction, metric, or threshold;
- reading Phase 47 outputs to rank or configure a candidate;
- overwriting E01–E09 artifacts or any V1 artifact/signoff;
- selecting a candidate, seed, loss weight, architecture, feature threshold, or ensemble weight from Test behavior;
- silently dropping target IDs from one side of a comparison;
- fitting feature statistics or scalers outside the fold-train population;
- reusing optimizer or scheduler state across independent runs;
- launching expensive training without an exact command and explicit Human approval.

Every failed or interrupted run remains recorded. No failed evidence may be silently removed.

## 4. Common experiment contract

Unless the named experiment explicitly changes a field, every candidate preserves:

- the current Human-accepted incumbent at experiment entry;
- RO1/RO2/RO3 definitions and ordered target IDs;
- WB0 context carry-over and horizon 1;
- fold-local X and Y scaling;
- seed 42 for screening;
- identical DataLoader order and worker-seeding policy;
- fresh model initialization and fresh optimizer/scheduler state;
- AdamW, LR `3e-4`, weight decay `1e-3`, scheduler OFF;
- batch size 32, MSE, gradient clip 1.0, max epochs 50;
- current Stage-A early stopping and exact-best-epoch Stage-B refit;
- pooled rolling-origin RMSE as primary selection metric;
- pooled MAE, worst-fold RMSE, fold consistency, and fold-train-defined rapid-change metrics as diagnostics/guardrails;
- `Test = NOT_ACCESSED`.

One primary change is allowed per OFAT experiment. E14 is the only planned exception: it is a separately declared, bounded multi-factor configuration-bundle search, not an OFAT claim.

Promotion is never automatic. A candidate must:

1. beat its population-matched control by the Human-locked pooled-RMSE tolerance;
2. satisfy the Human-locked MAE and worst-fold guardrails;
3. show the required fold consistency;
4. pass provenance, leakage, population, scaler, checkpoint, and artifact audits;
5. receive explicit Human promotion.

No weighted composite score is allowed.

## 5. Causal feature construction and population rules

### 5.1 Time convention

For a target `y[t+1]`, the final observable context row is `t`. Every target-derived feature at a context row `s <= t` may use only `y[s]` and earlier observations. It must never use `y[s+1]`, `y[t+1]`, a centered window, or a statistic fitted on future rows.

### 5.2 Canonical FS2_TF1 base order

The frozen 33-column base order is:

```text
lights, T1, RH_1, T2, RH_2, T3, RH_3, T4, RH_4, T5, RH_5,
T6, RH_6, T7, RH_7, T8, RH_8, T9, RH_9, T_out, Press_mm_hg,
RH_out, Windspeed, Visibility, Tdewpoint, Appliances, rv1, rv2,
hour_sin, hour_cos, dow_sin, dow_cos, weekend
```

New Wave 2 columns are appended in the exact order declared by the experiment. Existing columns are not reordered or removed.

### 5.3 Target-delta block

At every context row `s`:

```text
appliances_delta_1     = y[s] - y[s-1]
appliances_abs_delta_1 = abs(y[s] - y[s-1])
appliances_delta_2     = y[s] - y[s-2]
```

Proposed appended order is exactly the order above. Proposed feature count is `36` when added directly to FS2_TF1.

The `appliances_delta_2` definition above is a proposed Wave 2 lock and must be confirmed by Human review; it is not a second finite difference.

### 5.4 Past-only rolling-target block

At every context row `s`, trailing windows include the current observable value `y[s]` and only prior values:

```text
appliances_roll_mean_3
appliances_roll_mean_6
appliances_roll_mean_12
appliances_roll_std_6
appliances_roll_std_12
appliances_roll_max_12
appliances_roll_min_12
```

Rolling standard deviation uses population convention `ddof=0`. No partial window is permitted. The columns are appended in the order above. This block adds seven features.

### 5.5 Daily lag

At every context row `s`:

```text
Appliances_lag_144 = y[s-144]
```

This is a historical target only. Weekly lag 1008 is outside the current Wave 2 executable sequence.

### 5.6 Eligible target population

For L72 input windows, target-derived per-row features need history before the first row of each window. Therefore feature engineering can change eligibility:

- delta block requires two additional historical rows;
- rolling block requires eleven additional historical rows;
- lag144 requires 144 additional historical rows.

No imputation, zero filling, backfilling, partial rolling window, or future-derived filling is allowed merely to retain E01 targets.

Before each feature experiment, derive one ordered common target-ID intersection per fold. Both local control and challenger must be trained/evaluated on that exact intersection. Counts and SHA-256 population fingerprints must be locked before training. Because population changes are possible, historical E01 metrics must not be reused as the local control unless ordered target IDs are byte-for-byte identical.

### 5.7 Scaling lineage

Target-derived columns are created in raw Wh from pre-Test chronology before scaling. For every fold:

- X scaler fits the local fold-train rows only;
- Y scaler fits the local fold-train targets only;
- new continuous columns use the explicitly locked historical fold-local continuous-feature scaling behavior;
- scaler column order, fitted-row IDs, feature count, and bundle SHA-256 are recorded;
- Validation is transform-only;
- Test is never loaded.

Feature derivation has its own version, ordered feature fingerprint, causal audit, population fingerprint, and fold-local scaler lineage. A feature artifact from another order/count is incompatible.

## 6. Wave 2 experiment sequence

### E10 — Target-delta feature block

- Scientific question: do explicit local target changes improve the DIRECT Transformer without changing its architecture or objective?
- Control: current accepted incumbent re-materialized on the E10 common population.
- Primary change: append the three target-delta features in Section 5.3.
- Candidate scope: one local control plus one challenger; feature count `33 -> 36` if E01 is still incumbent.
- Expected run count: 12 total (`2 configs x 3 folds x Stage A/B`). Reuse is allowed only if the ordered population is proven identical; otherwise no control reuse.
- Required artifacts: feature contract, causal/no-future audit, ordered feature list/fingerprint, per-fold population IDs/counts/fingerprints, scaler audits, run registry, fold predictions/metrics, pooled comparison.
- Promotion: common contract in Section 4.
- Dependency: approved Wave 2 plan and locked delta definitions.
- Human training approval: required.

### E11 — Past-only rolling-target statistics

- Scientific question: do short causal level/volatility summaries add signal beyond the current incumbent?
- Control: Human-accepted incumbent after E10, re-materialized on the E11 common population.
- Primary change: append the seven rolling columns in Section 5.4.
- Candidate scope: one local control plus one challenger. If E10 is rejected, E11 does not inherit its delta block; if E10 is promoted, rolling features are appended to that promoted feature set.
- Expected run count: 12 total.
- Required artifacts: rolling formula/version contract, `ddof=0` lock, causal audit, ordered population equality, feature/scaler fingerprints, runs and comparison.
- Promotion: common contract in Section 4.
- Dependency: E10 result audit and Human keep/reject decision.
- Human training approval: required.

### E12 — Daily lag-144 feature

- Scientific question: does same-time previous-day consumption improve the incumbent after controlling for its smaller eligible population?
- Control: Human-accepted incumbent after E11, retrained on the lag144 common population.
- Primary change: append exactly `Appliances_lag_144`.
- Candidate scope: one population-matched local control plus one lag144 challenger.
- Expected run count: 12 total. Previous control metrics are not reusable unless ordered fold populations are identical, which is not expected for lag144.
- Required artifacts: lag causal audit, lost-target accounting, ordered common-population fingerprints, feature/scaler lineage, fold and pooled metrics.
- Promotion: common contract in Section 4.
- Dependency: E11 Human decision and lag144 population lock.
- Human training approval: required.

### E13 — Hybrid level-plus-delta loss

- Scientific question: can an auxiliary change objective reduce smoothing while preserving level accuracy?
- Control: accepted post-E12 incumbent with ordinary level MSE, reused read-only only when config and target population are exact.
- Primary change: loss policy only.
- Candidate scope: three challengers with `lambda_delta = {0.10, 0.25, 0.50}`.
- Exact loss in fold-local Y-standardized space:

```text
delta_true_model = (y_next_raw - y_context_raw) / y_scaler_scale
delta_pred_model = y_pred_model - y_context_model
L = MSE(y_pred_model, y_next_model)
    + lambda_delta * SmoothL1(delta_pred_model, delta_true_model, beta=1.0)
```

- Expected run count: 18 new runs (`3 challengers x 3 folds x Stage A/B`); control is reuse-only if exact.
- Required artifacts: algebra/unit audit, per-component epoch losses, config fingerprints, fold metrics, change diagnostics, pooled comparison.
- Promotion: RMSE plus guardrails; lower MAE alone cannot promote.
- Dependency: E12 Human decision and exact loss/scaling lock.
- Human training approval: required.

### E14 — Bounded staged multi-factor training search

- Scientific question: after feature/loss formulation is fixed, is there a useful interaction among LR, batch size, dropout, and clipping that OFAT missed?
- Control: accepted post-E13 incumbent, reuse-only if exact.
- Primary change: one pre-registered training-configuration bundle; this is the declared multi-factor exception, so no causal claim about any individual knob is allowed.
- Candidate scope: five challenger bundles, all AdamW, weight decay `1e-3`, scheduler OFF:

| Bundle | LR | Batch | Dropout | Clip |
|---|---:|---:|---:|---:|
| M1 | `2e-4` | 16 | 0.10 | 1.0 |
| M2 | `3e-4` | 16 | 0.05 | 1.0 |
| M3 | `5e-4` | 32 | 0.05 | 1.0 |
| M4 | `5e-4` | 64 | 0.10 | 1.0 |
| M5 | `3e-4` | 32 | 0.20 | 0.5 |

- Expected run count: 30 new runs (`5 challengers x 3 folds x Stage A/B`).
- Required artifacts: locked bundle table, equal-budget audit, parameter/runtime report, LR/gradient traces, runs, ranking, guardrails.
- Promotion: best eligible bundle by pooled RMSE after guardrails; no post-hoc bundle mutation.
- Dependency: feature and loss formulation frozen after E13.
- Human training approval: required.

### E15 — Moderate width/FFN capacity bundles

- Scientific question: does moderate width help after formulation and training settings are stable?
- Control: accepted post-E14 architecture (`d_model=64`, `heads=4`, `ffn=256`, `layers=2`).
- Primary change: proportional width bundle.
- Candidate scope:
  - `d_model=96`, `heads=4`, `ffn=384`, `layers=2`;
  - `d_model=128`, `heads=4`, `ffn=512`, `layers=2`.
- Expected run count: 12 new runs; exact control reused read-only.
- Required artifacts: parameter counts (<1 million), runtime/memory, initialization/config fingerprints, fold metrics and comparison.
- Promotion: common contract; capacity cannot promote on runtime-neutrality or one fold alone.
- Dependency: E14 Human decision.
- Human training approval: required.

### E16 — Depth expansion

- Scientific question: does one additional encoder block improve the selected width?
- Control: accepted post-E15 model at two layers.
- Primary change: `num_layers: 2 -> 3` only.
- Candidate scope: one three-layer Post-LN challenger; all other capacity fields fixed.
- Expected run count: 6 new runs; control reuse-only if exact.
- Required artifacts: parameter/runtime/gradient diagnostics, fold metrics and pooled comparison.
- Promotion: common contract.
- Dependency: E15 Human decision.
- Human training approval: required.

### E17 — Pre-LN ablation

- Scientific question: at the selected deeper architecture, does Pre-LN improve stability and development performance?
- Control: selected three-layer Post-LN model.
- Primary change: norm order `POST_NORM -> PRE_NORM` only.
- Candidate scope: one Pre-LN challenger.
- Expected run count: 6 new runs; control reuse-only if exact.
- Required artifacts: strict architecture/config report, gradient/LR traces, parameter-count equality, fold metrics and comparison.
- Promotion: common contract.
- Dependency: E16 must promote or explicitly justify a deeper-model stability test. If E16 is rejected and no deeper model is approved, E17 is deferred.
- Human training approval: required.

### E18 — Fine-tuning eligibility gate

- Scientific question: is a staged warm-start experiment scientifically comparable and technically compatible with the selected Wave 2 architecture?
- Control: scratch-trained exact architecture on the same folds/population.
- Primary change: none; this is a read-only gate.
- Candidate count: 0.
- Expected run count: 0.
- Required artifacts: checkpoint provenance table, per-fold source mapping, checksum verification, architecture/input/head compatibility, selection-influence audit, scratch-control proof.
- Dependency: Wave 2 architecture/features/head must be frozen after E17.
- Human training approval: not applicable to the gate. FT-A/B/C need a separate approved plan if and only if this gate passes.

Conditional fine-tuning order, not currently executable:

```text
FT-A: verified per-fold encoder checkpoint -> new head -> freeze encoder and input projection -> fresh optimizer -> head only
FT-B: unfreeze top encoder block -> fresh parameter groups -> fresh optimizer/scheduler state
FT-C: full encoder unfreeze -> predeclared differential LR -> fresh optimizer/scheduler state
```

Each stage must compare with the exact architecture trained from scratch. Failure at any stage closes later stages.

### E19 — Conditional TCN baseline

- Scientific question: if the Transformer plateaus, does a causal convolutional inductive bias perform better under the same development protocol?
- Control: current Human-accepted Transformer incumbent, reused read-only if feature population/loss match exactly.
- Primary change: model family `TRANSFORMER_ENCODER -> TCN`.
- Candidate scope: one reference causal TCN with four residual blocks, 64 channels, kernel size 3, dilations `[1,2,4,8]`, dropout 0.1, last-step readout, and the incumbent input features/loss/training budget.
- Expected run count: 6 new runs.
- Required artifacts: receptive-field proof, causal-padding test, parameter/runtime report, fold metrics and comparison.
- Promotion: common contract.
- Dependency: TCN trigger in Section 10 and separate Human approval.
- Human training approval: required.

### E20 — Matched three-seed finalist confirmation

- Scientific question: does one Human-selected finalist retain its development advantage across seeds rather than benefiting from seed 42?
- Control: E01-equivalent matched baseline.
- Primary change: seed replication of already locked configurations; no hyperparameter selection is permitted.
- Candidate scope: one finalist and its matched E01 control at seeds `42, 123, 2026`.
- Expected run count: up to 36 total evidence runs (`2 configs x 3 seeds x 3 folds x Stage A/B`). If exact seed-42 control/finalist evidence passes checksum/config/population audits, those 12 runs are reused and 24 new runs remain.
- Required artifacts: per-seed/fold registry, seed-level and pooled metrics, mean/SD, paired improvement table, provenance and leakage signoff.
- Promotion: final Human review using paired seed evidence; never choose the best seed.
- Dependency: one finalist selected after E17/E19 and all numeric guardrails locked.
- Human training approval: required.

## 7. Experiment order and stop gates

```text
E10 target delta
-> Human result review
-> E11 past-only rolling statistics
-> Human result review
-> E12 lag144 with common-population control
-> Human result review
-> E13 level-plus-delta loss
-> Human result review
-> E14 bounded multi-factor search
-> Human result review
-> E15 moderate width
-> E16 depth
-> E17 Pre-LN when eligible
-> E18 fine-tuning eligibility gate
-> E19 TCN only if triggered
-> Human finalist selection
-> E20 matched three-seed confirmation
```

No downstream experiment inherits a rejected upstream factor. If a candidate is rejected, the next experiment starts from the latest Human-accepted incumbent and constructs its own population-matched control.

## 8. Loss experiment order

Primary Wave 2 loss work is E13 level-MSE plus delta-SmoothL1. Existing MSE is the control. The same historical Huber configuration is not rerun.

Only if E13 fails specifically on pooled MAE while retaining credible change diagnostics may a later separately planned level-loss hybrid be proposed:

```text
alpha * MSE(level) + (1-alpha) * SmoothL1(level), alpha in {0.50, 0.75, 0.90}
```

Change-aware sample weighting and direction auxiliary loss remain deferred. They require fold-train-derived thresholds and a new Human-approved plan; they are not automatically executable Wave 2 experiments.

## 9. Fine-tuning audit

### Current checkpoint candidates

Only fold-matched E01 Stage-B development refit checkpoints are plausible current encoder sources:

| Fold | Run ID | Epoch | SHA-256 |
|---|---|---:|---|
| RO1 | `RUN_V2_TR_E01_RO1_B_0004_533B5F33` | 2 | `6aa201d63388c1922c5190c1869b640e945dc2a95c6a03a69f3307912ba478a3` |
| RO2 | `RUN_V2_TR_E01_RO2_B_0005_19A72D17` | 8 | `47a8588f84ad8b1143c317dafe76830c1bfa56aa0ff1e70227b7bbd070df6140` |
| RO3 | `RUN_V2_TR_E01_RO3_B_0006_AA5D66A6` | 21 | `3f85415a4dc157a552b19bab6c17334420d307cb5b398139a14e5c293312981b` |

The files were not deserialized for this audit. Their run metadata reports `COMPLETED`, exact-epoch refit, seed 42, no warm start, and no Stage-B validation.

### Audit result

- Provenance status: provisionally verified at path/run/status/SHA level; state-dict compatibility is not verified because weights were not loaded.
- Architecture compatibility: exact for the E01 Transformer (`d_model=64`, 2 layers, 4 heads, FFN 256, Post-LN, LAST_STEP).
- Feature compatibility: exact only for the frozen 33-column FS2_TF1 order. Any promoted delta/rolling/lag feature changes input projection count/order and blocks direct reuse of the E01 input projection.
- Test influence: E01 checkpoint selection used the pre-Test rolling-origin development protocol; its manifest reports `NOT_ACCESSED`. Phase47 checkpoints are forbidden.
- Scratch control availability: E01 provides scratch DIRECT control for the exact E01 architecture. A future Wave 2 feature/head/capacity architecture needs its own exact scratch control before warm-start comparison.
- Input projection reuse: not allowed when feature count/order differs; partial encoder-only transfer would be a different, separately specified experiment.

Current disposition:

```text
FINETUNING_STATUS = DEFERRED
```

Fine-tuning becomes READY only when E18 proves exact per-fold source mapping, state-dict compatibility, identical feature projection, identical architecture, no Test influence, and an exact scratch control. Otherwise it remains BLOCKED and FT-A/B/C are not scheduled.

## 10. TCN trigger

E19 may open only if, after E17 and Human review, one of the following is true:

1. no Transformer Wave 2 challenger is promoted;
2. the best Transformer improvement is below the Human-locked meaningful tolerance;
3. a nominal RMSE improvement fails MAE, worst-fold, fold-consistency, provenance, or leakage guardrails;
4. prediction/change diagnostics still demonstrate a Human-confirmed plateau.

TCN does not open merely because it appears in the source plan. It requires an explicit Human trigger decision and implementation plan.

## 11. Finalist and three-seed protocol

After all opened Wave 2 experiments:

1. freeze the complete development evidence table;
2. select exactly one finalist by Human review, not automatically;
3. verify config, feature order, population, scalers, checkpoint lineage, parameter count, runtime, failure ledger, and leakage status;
4. lock the finalist before E20;
5. run matched seeds `42, 123, 2026` on the same RO folds for finalist and control;
6. reuse seed-42 evidence only after exact checksum/config/population equivalence;
7. report each seed and fold, mean/SD, pooled metrics, worst fold, and paired deltas;
8. never select the best seed or retune from seed results;
9. return to Human review for final promotion.

Test remains inaccessible throughout finalist selection and seed confirmation.

## 12. Expected Wave 2 namespace and artifacts

Every experiment uses an isolated V2 namespace:

```text
COURSE_WORK/artifacts/model_improvement_v2/experiments/E10/
...
COURSE_WORK/artifacts/model_improvement_v2/experiments/E20/
```

Each executable experiment requires:

- immutable config snapshot and config fingerprint;
- source-control provenance and control artifact checksums;
- feature/loss/model contract as applicable;
- common-population and Test-firewall audit;
- experiment-specific registry namespace and run IDs;
- Stage-A and Stage-B run records/checkpoints/histories;
- Stage-C predictions and fold metrics;
- pooled metrics, guardrails, ranking, and Human decision field;
- execution manifest and no-retrain recovery contract;
- `test_status: NOT_ACCESSED`.

Separate ownership remains required for:

```text
MODEL_CONFIG
DEVELOPMENT_TRAINING_CONFIG
FINAL_REFIT_CONFIG
EVALUATION_CONFIG
```

## 13. Risks and hard stops

Hard stop on:

- any Test access or Test-derived choice;
- mutation of V1 or E01–E09 scientific evidence;
- target-derived feature using a future value;
- feature order/count or input projection mismatch;
- unequal target populations in a comparison;
- global rather than fold-local fitting;
- silent imputation to preserve population;
- more than one change outside the declared E14 bundle search;
- warm-start without exact provenance and scratch control;
- reuse of optimizer/scheduler state;
- missing numeric promotion/guardrail locks at training time;
- failed focused/static validation;
- training without an exact command and Human approval.

## 14. Definition of ready for Wave 2 implementation

Wave 2 implementation preparation may begin only after Human review confirms:

```text
WAVE2_PLAN = ACCEPTED
E01_GLOBAL_BASELINE = ACCEPTED
E10_DELTA_DEFINITIONS = ACCEPTED
FEATURE_ORDER_AND_POPULATION_POLICY = ACCEPTED
WAVE2_PROMOTION_RMSE_TOLERANCE = LOCKED
WAVE2_POOLED_MAE_TOLERANCE = LOCKED
WAVE2_WORST_FOLD_TOLERANCE = LOCKED
WAVE2_FOLD_CONSISTENCY_RULE = LOCKED
TEST_ACCESS = NO
TRAINING_AUTHORIZED = NO
```

Recommended first implementation task is E10 only. E11–E20 remain planned, not automatically authorized.

## 15. Final handoff

```text
Wave 2 experiments
-> Human review
-> finalist selection
-> matched 3-seed confirmation
-> documentation/notebook/artifact update
-> final validation
-> Human-authorized commit/push on CourseWorkV12
```

No old-Test rerun is part of this plan. Any later evaluation on the old Test requires a separately justified protocol and must be labelled `POST_HOC_V2_BENCHMARK`; a genuinely unbiased final evaluation requires new unseen data.
