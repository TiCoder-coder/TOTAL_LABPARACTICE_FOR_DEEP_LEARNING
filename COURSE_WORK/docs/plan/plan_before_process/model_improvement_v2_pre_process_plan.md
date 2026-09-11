# MODEL_IMPROVEMENT-v2 — Official Pre-Process and Review Plan

## 1. Objective

This document is the official pre-process and review gate for `MODEL_IMPROVEMENT-v2`.

Its objectives are to:

- preserve the complete V1 scientific record without overwrite or silent repair;
- lock one canonical development baseline for every V2 promotion decision;
- define only the first-wave experiments `E00` through `E09`;
- enforce controlled comparisons with one primary experimental change at a time;
- prohibit Phase 47 Test evidence from V2 tuning and selection;
- require explicit Human approval before any expensive training command is executed;
- separate V2 configuration, source, registry, run and artifact ownership from V1.

This document authorizes planning and later implementation preparation only. It does not authorize training, inference, Test access, artifact regeneration, checkpoint loading, or modification of frozen V1 evidence.

## 2. Source Plan and Scope

Authoritative source plan:

```text
COURSE_WORK/docs/Plan_improve_model.md
```

Improvement track:

```text
MODEL_IMPROVEMENT-v2
```

This pre-process plan narrows the source plan to Wave 1. It does not modify the source plan and does not authorize all proposals described there.

Wave 1 contains only:

```text
E00 — V1 lineage classification and freeze
E01 — V1-equivalent Phase 44 development-baseline reproduction
E02 — FS1_TF1 versus FS2_TF1
E03 — DIRECT versus RESIDUAL-to-Persistence
E04 — residual LINEAR head versus MLP head
E05 — residual gate OFF versus ON
E06 — AdamW learning-rate audit
E07 — scheduler audit
E08 — plain SGD bounded audit
E09 — SGD with momentum bounded audit
```

After `E09`, execution must stop at a Human review gate. Wave 2 must not open automatically.

## 3. Frozen V1 Policy

V1 Phase 45, Phase 46 and Phase 47 are frozen historical evidence.

Mandatory policy:

- do not overwrite V1 configs, checkpoints, scalers, predictions, reports, signoffs, handoffs, manifests, analyses or conclusions;
- do not silently repair V1 provenance during V2 work;
- preserve and document known provenance debt;
- classify incompatible active/historical lineages explicitly rather than merging them;
- do not promote a V1 artifact into V2 merely because it has a `PASS` field;
- require checksum, configuration, scaler, population, seed and checkpoint-lineage verification before any permitted reuse;
- keep all V2 outputs in an isolated namespace.

Known provenance debt includes, at minimum:

- inconsistency between corrected active Phase 46 run references and Phase 47 historical run/checkpoint references;
- Phase 45 lookback/fingerprint consistency concerns;
- Phase 43 LSTM lookback, feature-set and scaler-lineage inconsistencies;
- mixed development and final-refit fields in current configuration artifacts.

V1 provenance debt is not authorization to mutate V1. Any future repair requires a separate issue report, repair plan and Human approval.

## 4. Canonical V2 Development Baseline

The following baseline is locked for V2 development comparison:

```yaml
candidate_id: TR_C2_ALT_LOOKBACK
config_fingerprint: 585c5e79e6a1c8c49efdd2f7767e6d3d4c628835acfda360a2fa66a2ef5c1e24
feature_set: FS2_TF1
lookback: 72
boundary_protocol: WB0_CONTEXT_CARRY_OVER
rolling_origin_protocol: RO3_EXPANDING_PRETEST-v1
folds:
  - RO1
  - RO2
  - RO3
fold_count: 3
fold_local_scaling: true
baseline_pooled_rmse_wh: 59.85979463604273
phase44_status: PASS
phase44_test_status: NOT_ACCESSED
optimizer: AdamW
learning_rate: 3e-4
loss: MSE
```

This is a development baseline derived from Phase 44 pre-Test rolling-origin evidence. It is not a Phase 47 Test baseline.

Every V2 promotion decision must compare against this development baseline or a faithfully reproduced V1-equivalent baseline on identical folds and target populations.

Phase 47 Test metrics must not be used for V2 selection, ranking, thresholds, stopping criteria or hyperparameter choice.

## 5. Data and Test Firewall

Allowed development evidence:

- Train chronology;
- Validation chronology;
- the locked `RO1`, `RO2`, `RO3` expanding pre-Test folds;
- fold-train-derived statistics;
- fold-validation metrics computed only after the corresponding fold model is fixed.

Forbidden during Wave 1:

- Test target access;
- new Test inference;
- reading Phase 47 Test metrics for candidate comparison or threshold setting;
- choosing LR, optimizer, architecture, feature set, scheduler, fine-tuning policy, seed or candidate from known Phase 47 results;
- deriving feature thresholds, regime thresholds, scalers or normalization statistics from Test;
- using Test-derived post-hoc ensemble or blend weights;
- relabeling the old Test as unseen or unbiased V2 evidence.

Each fold must fit its own X scaler and Y scaler using fold-train rows only. Any regime or diagnostic threshold must also be derived from fold-train only.

If code, configuration, notebook, cached artifact or command would access Test during Wave 1, the mandatory action is:

```text
STOP — REPORT — WAIT FOR HUMAN DECISION
```

## 6. Experiment Principles

All controlled comparisons require:

- the same `RO1`, `RO2`, `RO3` definitions;
- identical target populations between baseline and candidate within each comparison;
- the same seed policy;
- the same model-initialization policy;
- the same data-order and DataLoader-seeding policy;
- the same maximum optimizer-step or epoch budget where applicable;
- the same loss unless loss is the explicitly approved primary variable;
- the same gradient clipping;
- fresh optimizer state;
- fresh scheduler state;
- fold-local scalers only;
- no Test access;
- one primary experimental change at a time;
- full recording of completed, failed, interrupted and rejected runs.

No candidate may receive a hidden fallback. A failed run must remain in the V2 registry with its failure class and available diagnostics.

## 7. Wave 1 Experiment Matrix E00–E09

### E00 — V1 Lineage Classification and Freeze

| Field | Contract |
|---|---|
| Training | Forbidden |
| Objective | Classify and freeze V1 Phase45/46/47 lineages without modifying them |
| Primary change | None; read-only governance step |
| Inputs | Active V1 configs, fingerprints, signoffs, handoffs and referenced checkpoint metadata |
| Required output | V1 baseline snapshot and lineage classification under the V2 artifact namespace |
| Pass condition | Every preserved lineage is internally described; unresolved contradictions remain explicit |
| Hard stop | Any proposed mutation, regeneration, inference or silent repair of V1 |

### E01 — Reproduce V1-Equivalent Phase 44 Development Baseline

| Field | Contract |
|---|---|
| Objective | Reproduce the locked Phase 44 development baseline under the same pre-Test protocol |
| Baseline | `TR_C2_ALT_LOOKBACK`, fingerprint `585c5e...e24` |
| Primary change | None; reproduction only |
| Folds | `RO1`, `RO2`, `RO3` |
| Feature set | `FS2_TF1` |
| Lookback/boundary | 72 / `WB0_CONTEXT_CARRY_OVER` |
| Optimizer/LR/loss | AdamW / `3e-4` / MSE |
| Target result | Pooled development RMSE `59.85979463604273 Wh`, subject to exact lineage and aggregation verification |
| Hard stop | Population, scaler, fold, config or aggregation mismatch |

E01 does not authorize automatic training. Implementation preparation, validation and an exact Human-run command are required first.

### E02 — Feature Variant Ablation

| Field | Contract |
|---|---|
| Objective | Test whether random controls provide robust development benefit |
| Primary variable | Feature variant only |
| Candidates | `FS1_TF1`, `FS2_TF1` |
| Fixed | Model, initialization, folds, target population, optimizer, LR, loss, clipping and budget |
| Required audit | Feature order, feature count, scaler ownership and common-population equality |
| Promotion evidence | Development rolling-origin evidence only |

`rv1` and `rv2` remain negative controls. They must not be retained or removed using Phase 47 Test behavior.

### E03 — Prediction Formulation Ablation

| Field | Contract |
|---|---|
| Objective | Test explicit Persistence inductive bias |
| Primary variable | Prediction formulation only |
| Candidates | `DIRECT`, `RESIDUAL_TO_PERSISTENCE` |
| Fixed | Selected E02 feature variant, linear head, model capacity, optimizer, LR, scheduler state, loss and folds |
| Residual identity | `y_hat(t+1) = y(t) + delta_hat(t+1)` |
| Required tests | Zero delta equals exact Persistence; correct Y-scaler space; no target-row leakage; output shape `[B,1]` |

The most recent observed `Appliances` value must be transformed into target model space using a verified Y-scaler rule. The X-scaled input channel must not be silently treated as an equivalent persistence base.

### E04 — Residual Regression-Head Ablation

| Field | Contract |
|---|---|
| Objective | Test whether a nonlinear residual head improves correction magnitude |
| Primary variable | Regression head only |
| Candidates | `LINEAR`, `MLP` |
| Fixed | Residual formulation, feature variant, encoder capacity, optimizer, LR, scheduler, loss and folds |
| Required audit | Equal input representation and parameter-count/runtime reporting |

### E05 — Learned Residual-Gate Ablation

| Field | Contract |
|---|---|
| Objective | Test sample-adaptive use of the neural correction |
| Primary variable | Learned gate only |
| Candidates | `GATE_OFF`, `GATE_ON` |
| Fixed | Selected residual head, features, encoder, optimizer, LR, scheduler, loss and folds |
| Gated identity | `y_hat(t+1) = y(t) + sigmoid(gate_logits) * delta_hat(t+1)` |
| Required diagnostics | Gate distribution, saturation, fold consistency and prediction equivalence when gate approaches zero |

No Test-derived regime rule may be hard-coded into the gate.

### E06 — AdamW Learning-Rate Audit

| Field | Contract |
|---|---|
| Objective | Audit lower constant learning rates for the selected E05 formulation |
| Primary variable | Learning rate only |
| Optimizer | AdamW |
| Candidate LR | `5e-5`, `1e-4`, `2e-4`, `3e-4` |
| Scheduler | OFF |
| Fixed | Architecture, feature set, batch size, loss, clipping, weight decay, folds and budget |
| Required logging | Epoch, training loss, development metrics, LR, gradient norm and clipped-batch fraction |

### E07 — Scheduler Audit

E07 may begin only after E06 result audit and Human approval.

| Field | Contract |
|---|---|
| Objective | Compare the selected constant-LR control against scheduler variants |
| Primary variable | Scheduler policy only |
| Control | Selected E06 constant-LR setting |
| Initial candidates | Constant control, warmup-plus-cosine, ReduceLROnPlateau |
| Exact scheduler values | `HUMAN_TO_LOCK_BEFORE_TRAINING` |
| Fixed | Optimizer, selected maximum/initial LR where comparable, architecture, features, folds, loss, clipping and optimizer-step budget |
| Required audit | Scheduler step timing, LR trace, early-stopping interaction and equal-budget comparison |

Scheduler configuration must not be inferred from Phase 47 Test results. Exact warmup ratio, minimum LR, plateau factor and patience require a later Human lock before training.

### E08 — Plain SGD Bounded Audit

| Field | Contract |
|---|---|
| Objective | Perform a bounded optimizer ablation requested by the team |
| Primary variable | Optimizer/LR candidate within the predeclared SGD experiment family |
| Optimizer | `torch.optim.SGD` |
| Momentum | `0` |
| Candidate LR | `1e-4`, `3e-4`, `1e-3`, `3e-3` |
| Scheduler | OFF initially |
| Nesterov | OFF |
| Fixed | Selected architecture/features, folds, loss, clipping, model initialization, data order and budget |
| Comparator | Fair AdamW constant-LR control |

### E09 — SGD With Momentum Bounded Audit

| Field | Contract |
|---|---|
| Objective | Test SGD with momentum separately from plain SGD |
| Primary variable | Optimizer configuration within the predeclared momentum-SGD family |
| Optimizer | `torch.optim.SGD` |
| Momentum | `0.9` |
| Candidate LR | `1e-3`, `3e-3`, `1e-2` |
| Weight decay | `0`, `1e-4` |
| Scheduler | OFF initially |
| Nesterov | OFF initially |
| Fixed | Selected architecture/features, folds, loss, clipping, initialization, data order and budget |
| Comparator | Fair AdamW control under an equivalent scheduler state |

After E09:

```text
MANDATORY HUMAN REVIEW GATE
WAVE_2_AUTO_OPEN = false
```

## 8. SGD Experiment Policy

SGD is a bounded optimizer ablation requested by the team.

It is not assumed to be better than AdamW.

Plain SGD and SGD with momentum are separate experiments. Their results, run IDs and failure states must remain separate.

Comparing tuned SGD plus scheduler against untuned AdamW is forbidden. If an SGD scheduler is tested later, AdamW must receive a fair equivalent scheduler comparison using the same folds, initialization policy, data order, budget, loss and clipping.

Independent SGD candidates require fresh optimizer and scheduler state. AdamW state, momentum buffers or scheduler state must not be transferred into SGD.

## 9. LR and Scheduler Policy

E06 is the only Wave 1 LR search before scheduler evaluation:

```text
AdamW
scheduler = OFF
LR = {5e-5, 1e-4, 2e-4, 3e-4}
```

E07 must use the selected constant-LR result as its control. Scheduler variants must use predeclared step timing and comparable training budgets.

The following must be locked before E07 training:

```text
warmup_ratio: HUMAN_TO_LOCK_BEFORE_TRAINING
minimum_lr: HUMAN_TO_LOCK_BEFORE_TRAINING
cosine_duration_or_T_max: HUMAN_TO_LOCK_BEFORE_TRAINING
plateau_factor: HUMAN_TO_LOCK_BEFORE_TRAINING
plateau_patience: HUMAN_TO_LOCK_BEFORE_TRAINING
early_stopping_patience: HUMAN_TO_LOCK_BEFORE_TRAINING
early_stopping_min_delta: HUMAN_TO_LOCK_BEFORE_TRAINING
```

No scheduler parameter may be selected using Test evidence.

## 10. Seed and Compute Policy

Screening seed:

```text
42
```

Finalist seeds only:

```text
42
123
2026
```

Do not initially run every candidate across all seeds. Three-seed evaluation is reserved for a Human-approved finalist stage.

Every failed, interrupted, timed-out or non-finite run must remain recorded. It must not be silently deleted, replaced or excluded from reporting.

The agent must not autonomously launch expensive training.

For every expensive run, the exact terminal command, working directory, environment, expected outputs and estimated scope must be printed for the Human first.

Mandatory workflow:

```text
PLAN
→ IMPLEMENTATION PREP
→ STATIC/UNIT VALIDATION
→ PRINT EXACT TERMINAL TRAIN COMMAND
→ HUMAN APPROVAL
→ HUMAN RUNS COMMAND
→ RESULT AUDIT
→ NEXT EXPERIMENT
```

## 11. Promotion Policy

Primary metric:

```text
pooled rolling-origin RMSE
```

Mandatory guardrails:

- pooled MAE;
- worst-fold RMSE;
- fold consistency;
- rapid-change metrics where relevant and derived under fold-train thresholds;
- provenance PASS;
- leakage audit PASS;
- identical target populations for candidate and baseline.

No weighted composite score is permitted.

Test metrics must not define promotion thresholds.

Exact numeric improvement tolerance:

```text
HUMAN_TO_LOCK_BEFORE_TRAINING
```

Exact pooled-MAE tolerance:

```text
HUMAN_TO_LOCK_BEFORE_TRAINING
```

Exact worst-fold tolerance and required fold-consistency rule:

```text
HUMAN_TO_LOCK_BEFORE_TRAINING
```

Until these values are locked, experiments may not be promoted and scientific training must not start.

## 12. Human Training Gate

This plan does not authorize training.

Before each experiment that requires training, all of the following must be present:

1. approved implementation scope;
2. exact experiment configuration;
3. one-primary-change comparison statement;
4. verified folds and common target population;
5. fold-local scaler contract;
6. seed and data-order contract;
7. fresh optimizer/scheduler-state assertion;
8. static and focused unit-test results;
9. exact terminal command printed for review;
10. explicit Human approval for that command.

The Human runs the expensive command. The agent may audit resulting artifacts only after the command completes and the Human requests result audit.

Missing any gate yields:

```text
TRAINING_AUTHORIZED = NO
STOP
```

## 13. Wave 2 Deferred Work

The following remain candidates only and are not immediately executable under this plan:

- target delta features;
- rolling target statistics;
- lag-144;
- hybrid or delta loss;
- capacity expansion;
- Pre-LN;
- TCN;
- PatchTST;
- iTransformer;
- TSMixer, PatchTSMixer, TimeMixer or other alternative architectures;
- fine-tuning, checkpoint warm-start, freezing or unfreezing.

Wave 2 requires:

```text
E09 result audit
→ Human review
→ separate Wave 2 pre-process plan
→ Human approval
```

## 14. Expected V2 File and Artifact Namespace

Expected configuration ownership:

```text
COURSE_WORK/configs/improvement_v2/
├── model_improvement_contract.json
├── model_config.json
├── development_training_config.json
├── final_refit_config.json
├── evaluation_config.json
├── search_space.json
└── promotion_policy.json
```

Required configuration separation:

```text
MODEL_CONFIG
DEVELOPMENT_TRAINING_CONFIG
FINAL_REFIT_CONFIG
EVALUATION_CONFIG
```

Expected source namespace:

```text
COURSE_WORK/src/course_work/improvement_v2/
COURSE_WORK/src/course_work/models/transformer_v2.py
COURSE_WORK/src/course_work/models/residual_forecasting.py
COURSE_WORK/src/course_work/models/regression_heads.py
COURSE_WORK/src/course_work/training/schedulers.py
COURSE_WORK/src/course_work/evaluation/improvement_metrics.py
COURSE_WORK/src/course_work/experiments/improvement_v2.py
```

Expected artifact namespace:

```text
COURSE_WORK/artifacts/improvement_v2/
├── v1_baseline_snapshot.json
├── contracts/
├── lineage/
├── runs/
├── experiments/E00/
├── experiments/E01/
├── experiments/E02/
├── experiments/E03/
├── experiments/E04/
├── experiments/E05/
├── experiments/E06/
├── experiments/E07/
├── experiments/E08/
├── experiments/E09/
├── rolling_origin/
├── registry/
└── reports/
```

No V2 output may be written into V1 Phase 0–59 canonical artifact directories.

## 15. Risks and Hard Stops

### Risks

- Test leakage through direct data access, cached predictions, thresholds or result-driven choices;
- stale Phase 45/46/47 artifacts treated as one coherent lineage;
- wrong checkpoint or scaler reuse;
- L36/L72 lookback and population mismatch;
- FS1/FS2 feature-order or scaler mismatch;
- optimizer or scheduler state reused across independent experiments;
- unequal initialization, data order, training budget or fold population;
- failed runs removed from the registry;
- multiple primary changes combined in one comparison;
- automatic expansion into Wave 2;
- expensive training launched without Human approval.

### Hard stops

Execution must stop when any of the following occurs:

- any Test access is requested or detected;
- a V1 frozen file would be overwritten or silently repaired;
- lineage, checkpoint, scaler, feature order, lookback or target population is unresolved;
- fold-local scaling cannot be verified;
- candidate and baseline populations differ;
- more than one primary variable changes without a separately approved factorial design;
- promotion tolerances remain unlocked at the training boundary;
- static or focused tests fail;
- a run fails and the proposed response is to hide or delete it;
- an expensive training command lacks explicit Human approval;
- E09 is complete and Wave 2 has not received a new approved plan.

## 16. Definition of Ready for Part 3

Part 3 means implementation preparation only, not training.

The project is ready for Part 3 only when the Human explicitly approves this pre-process plan and confirms the remaining locks below:

```text
V2_NAMESPACE_POLICY = ACCEPTED
V1_FROZEN_POLICY = ACCEPTED
CANONICAL_DEVELOPMENT_BASELINE = ACCEPTED
WAVE1_E00_TO_E09 = ACCEPTED
PROMOTION_RMSE_TOLERANCE = HUMAN_TO_LOCK_BEFORE_TRAINING
PROMOTION_MAE_TOLERANCE = HUMAN_TO_LOCK_BEFORE_TRAINING
WORST_FOLD_TOLERANCE = HUMAN_TO_LOCK_BEFORE_TRAINING
FOLD_CONSISTENCY_RULE = HUMAN_TO_LOCK_BEFORE_TRAINING
```

The numeric promotion locks may be finalized during Part 3 implementation preparation, but all must be locked before any training command is approved.

Current state at document creation:

```text
PRE_PROCESS_PLAN_CREATED = true
BASELINE_DOCUMENT_LOCKED = true
TRAINING_AUTHORIZED = false
TEST_ACCESS_AUTHORIZED = false
WAVE2_AUTHORIZED = false
READY_FOR_PART3 = WAITING_FOR_HUMAN_APPROVAL
```
