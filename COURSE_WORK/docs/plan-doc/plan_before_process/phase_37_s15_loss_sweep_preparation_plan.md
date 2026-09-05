# Phase 37 — S15 Loss Sweep Pre-Process Plan

## Approval state

- Phase ID: `37`
- Phase name: `S15 Loss Sweep`
- Plan ID: `CW-PHASE37-S15-PREPARATION-001`
- Execution status: `HUMAN_APPROVED_PREPARATION_IMPLEMENTED`
- Human approval: granted for preparation implementation on `2026-08-24`.
- This document authorizes only the approved preparation implementation, focused verification, audit-only, dry-run and preflight. It does not authorize official Huber training, MSE retraining, Test access, winner selection or Phase 38 execution.

## Sources of truth and alignment

The preparation is governed by:

- `docs/plan-doc/plan_detail_for_each_phase/Phase_37_S15_Loss_sweep.md`
- `docs/plan-doc/plan_overview/Main_plan.md`
- `docs/RULE_BASE/rule_code.md`
- `docs/RULE_BASE/architecture_rule.md`
- `working_rule.md`
- the signed Phase 36 handoff under `artifacts/sweeps/S14_ffn/`

The overview, Phase Detail and Phase 36 handoff agree on the S15 comparison: MSE is the reused current reference and Huber with fixed `delta=1.0` in model-space is the only fresh condition. The requested scientific preparation is separately approval-gated because the architecture rule still treats Phase 37+ implementation as outside the current implemented scope until a Phase Detail, pre-process plan and Human approval all exist.

## Objective

Prepare, but do not execute, a Validation-only controlled comparison of training loss for the selected Transformer configuration. The only controlled scientific factor is `training_loss`. Full-precision verified Validation RMSE in Wh selects the winner. Raw MSE and Huber criterion magnitudes are never used to rank conditions.

## Scientific contract

- Sweep ID: `S15_LOSS`
- Sweep version: `SWEEP_S15_LOSS-v1`
- Controlled field: `training_loss`
- L0: `MSELoss(reduction="mean")`, `REUSE_REFERENCE`
- L1: `HuberLoss(delta=1.0, reduction="mean")`, `TRAIN_NEW`
- MSE reference run: `RUN_TR_S14_0023_A711A9B8`
- Huber execution: one fresh seed-42 run only
- Selection metric: verified full-precision Validation RMSE Wh
- Selection direction: minimum
- Exact tie: prefer MSE
- Test access: `FORBIDDEN`
- MSE retraining: `FORBIDDEN`
- Huber delta tuning: `FORBIDDEN`
- Huber official training during preparation: `FORBIDDEN`
- Phase 38 execution: `FORBIDDEN`

## Current repository state

### Canonical Phase 36 handoff

The signed Phase 36 evidence declares:

- overall status `PASS_WITH_WARNING`;
- `approved_for_phase37=true`;
- winner `F256` from `RUN_TR_S14_0023_A711A9B8`;
- selected `ffn_dim=256`;
- verified Validation RMSE `57.69679988114431 Wh`;
- condition policy `MSE=REUSE_REFERENCE`, `Huber=TRAIN_NEW`;
- Test status `FORBIDDEN`;
- inherited warning `H4_SOURCE_ARTIFACT_RETENTION_INCOMPLETE`.

The warning is non-critical but must be propagated through every applicable S15 manifest, winner, summary, report, Phase 38 reference update and sign-off. It must not be erased or converted into fabricated historical evidence.

### Existing shared infrastructure

- The global condition contract already reserves `L0` and `L1`.
- The Experiment Registry family list already declares `S15_LOSS` as Phase 37 and `PLANNED`.
- Phase 36 emits the required approved condition policy and signed handoff.
- The shared Transformer builder, Training Engine, Validation metrics, scaler loader, experiment registry, artifact utilities and previous sweep patterns exist.
- The canonical YS1 scaler and signed Phase 9 scaler evidence exist.

### Missing Phase-37-owned implementation

- No Phase 37 loss-sweep owner exists under `src/course_work/sweeps/`.
- Shared selective execution, recovery, finalization and reporting have no S15 owner registration.
- Terminal runners do not yet expose a guarded S15 Huber-only execution path.
- No focused Phase 37 loss-contract tests exist.
- The notebook has no Phase 37 presentation/orchestration section.
- No canonical `artifacts/sweeps/S15_loss/` scientific result set exists, as expected before execution.

Current implementation classification: `READY_FOR_VERIFICATION`; official Huber training remains separately Human-operated and was not executed by preparation.

## Dynamically resolved frozen configuration

The implementation must load the following from the signed Phase 36 handoff and its referenced run; reusable scientific code must not hard-code a winner label or run ID:

| Field | Current audited value |
|---|---:|
| Winner/reference run | `RUN_TR_S14_0023_A711A9B8` |
| Feature variant | `FS2_TF1` |
| Target scaling | `YS1` |
| Target model-space | standardized target |
| Lookback | `L36` |
| Pooling | `LAST_STEP` |
| Activation | `GELU` |
| Batch | `B32` / `32` |
| Learning rate | `0.0003` |
| Weight decay | `0.001` |
| Dropout | `0.1` |
| `d_model` | `64` |
| `num_heads` | `4` |
| `head_dim` | `16` |
| `num_layers` | `2` |
| `ffn_dim` | `256` |
| Current loss | `MSE` |
| Population version | `WINDOWPOP-v1` |
| Population fingerprint | `a40ded8802e90008535d268720bad9e9dcca5eee1436ddb359deea3ad39a1987` |
| Metric version | `METRICS-v1` |

Any mismatch between these audited expectations and dynamically loaded canonical evidence is a hard preflight block, not permission to patch the expectation into source.

## Huber delta and target-space contract

Hard protocol:

```text
delta_model_space = 1.0
delta_source = S15_PROTOCOL
delta_tuned = false
```

Because the selected target transform is currently `YS1`, the value is one standardized target unit, not one Wh. The approved implementation must call the canonical validated train-only target-scaler loader, validate the signed artifact checksum and resolve:

```text
delta_raw_wh_equivalent = frozen_y_scaler.scale_[0]
```

Current read-only canonical evidence reports:

- scaler path: `artifacts/scalers/y/YSCALER__YS1__SCALING-v1.joblib`;
- signed SHA256: `b3326a79da81b092460ef8d4a140a101b21f2433ff30c36971d626d2a2491697`;
- scaler scale and therefore expected raw-Wh equivalent delta: `106.853424078282 Wh`.

These values are audit expectations only. Reusable implementation must derive them at runtime from validated canonical evidence and must not pre-fill or silently accept them when the scaler cannot be loaded and verified.

## Criterion contract

The only allowed definitions are:

```python
torch.nn.MSELoss(reduction="mean")
torch.nn.HuberLoss(delta=1.0, reduction="mean")
```

Before criterion evaluation, require exact `[B, 1]` prediction/target shape equality, finite inputs and no broadcasting. Training loss operates in selected target model-space. The inverse target transform belongs only to Validation metric computation in Wh and must never enter the backward path.

SmoothL1 substitution, MAE, log-cosh, quantile, asymmetric or mixed objectives, loss scheduling, delta scheduling, residual/target clipping, sample weighting and criterion-dependent output activation are forbidden.

## Frozen-variable contract

Outside `training_loss`, keep identical:

- Train population, ordered Validation population, IDs, population fingerprint, feature order/fingerprint and loader policy;
- X scaler, target transform/scaler and target IDs;
- D/H/HD/N/F, sinusoidal PE, POST_NORM, mask policy, LAST_STEP pooling, GELU, dropout scope/probability and regression head;
- model parameter count, state-dict keys, parameter/buffer shapes and output shape;
- AdamW, LR, WD, batch, parameter groups, gradient clipping, scheduler, warmup, accumulation and mixed-precision policy;
- seed 42, Training Engine and METRICS-v1;
- maximum epochs 50, patience 10 and `min_delta=0`.

Early stopping and BEST checkpoint selection for both conditions remain Validation RMSE Wh. BEST means the earliest strict full-precision minimum.

## MSE reuse gate

`RUN_TR_S14_0023_A711A9B8` may be reused only after exact validation of the complete frozen contract, `MSELoss(reduction="mean")`, architecture and training fingerprints, population, metric/engine provenance, BEST evidence, and absence of Test access.

The MSE run must not be retrained. If retained initialization or sample-order fingerprints are absent, the respective audit must report `NOT_VERIFIABLE`; it must not infer equality or create replacement history.

## Fresh Huber policy

After all preparation gates pass and a Human separately runs the eventual terminal command, L1 must use:

- a newly registered Huber condition;
- reseed 42;
- fresh Train and Validation loaders under the frozen policy;
- a fresh Transformer instantiated from the dynamically resolved Phase 36 architecture;
- fresh `HuberLoss(delta=1.0, reduction="mean")`;
- fresh AdamW with the frozen LR/WD and parameter-group policy;
- no warm-start, checkpoint loading from MSE, weight copying, optimizer-state reuse or per-batch seed reset.

## Architecture and optimizer invariance audits

Preparation must instantiate disposable MSE/Huber model configurations without training and verify:

- equal architecture fingerprints;
- equal trainable parameter counts;
- equal state-dict key sets;
- equal parameter shapes and buffer shapes;
- equal output shapes;
- optimizer coverage of every trainable parameter exactly once;
- distinct model, criterion and training configuration fingerprints;
- criterion configuration is the only approved scientific delta.

Any model parameter, schema or optimizer-coverage difference blocks training authorization.

## Gradient, clipping and regime diagnostic support

Without changing training behavior, approved implementation must support recording mean/max pre-clip global gradient norm, clipped/total batches, clipping fraction and non-finite gradient events. The order remains `zero_grad → forward → criterion → backward → clip_grad_norm_(1.0, error_if_nonfinite=True) → optimizer.step`.

BEST Huber Validation diagnostics must support fractions of model-space residuals `|e| <= 1.0` and `|e| > 1.0`, plus median, p75, p90, p95 and maximum absolute residual. These are secondary diagnostics and never winner metrics. Test data is forbidden.

The optional first-batch gradient probe may exist only as a disposable isolated computation that cannot mutate or advance the official run.

## Loss-scale comparability contract

- Raw MSE versus raw Huber criterion values: `NOT_NUMERICALLY_CROSS_LOSS_COMPARABLE`.
- Validation RMSE Wh: cross-condition comparable and winner eligible.
- Validation MAE Wh: cross-condition comparable, secondary only.
- Validation R²: cross-condition comparable, secondary only.
- Huber regime, gradient and clipping diagnostics: secondary only.

No secondary metric or optimization diagnostic may override a lower verified Validation RMSE Wh.

## All-20-gate preflight

The approved implementation must audit the direct pre-training dependency path in one pass:

1. Phase 36 signed status and approval.
2. S14 winner identity and run lineage.
3. All prior selected factors dynamically frozen.
4. L0/L1 and S15 registry contracts.
5. Huber delta exactly 1.0.
6. Target model-space resolved.
7. Raw-Wh equivalent delta and scaler checksum resolved.
8. Same Train/Validation population.
9. Same target scaling.
10. Architecture equality.
11. Parameter-count and schema equality.
12. Exact criterion classes/reductions.
13. Shape equality and no broadcasting.
14. Training Engine selection metric remains Validation RMSE Wh.
15. Same optimizer, LR, WD, batch and clipping.
16. Initialization policy.
17. Sample-order policy.
18. MSE exact-reference reuse eligibility.
19. Test firewall.
20. Experiment Registry readiness for one fresh Huber run.

All blockers must be reported together before scientific execution. A failed hard gate leaves Huber `BLOCKED`.

## Focused test plan

After approval, focused tests must cover at least:

- exact MSE/Huber classes, mean reductions, positive/exact delta and zero trainable criterion parameters;
- zero-residual, quadratic, boundary, linear and positive/negative symmetry formulas;
- exact shape equality, broadcasting rejection and finite loss;
- identical architecture count/schema/shapes and criterion-only config delta;
- validated YS1 scaler loading and raw-Wh delta derivation;
- model-space backward and inverse-to-Wh Validation evaluation boundaries;
- Validation RMSE Wh early stopping/BEST selection and exact-tie MSE rule;
- frozen optimizer/loader/population contracts and no duplicate/missing optimizer parameters;
- MSE reuse-only behavior and fresh Huber preparation;
- warning propagation, Test firewall, output schemas and Phase 38 handoff policy.

Use exactly `/Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10`. Run only focused Phase 37 tests, audit-only, dry-run, preflight and a guarded Huber smoke that stops before `TrainingEngine.train()` if the established runner supports that boundary.

## Required O37.1–O37.41 output mapping

| Output | Required evidence / owner after scientific completion |
|---|---|
| O37.1–O37.4 | manifest, contract, preflight audit and run matrix |
| O37.5–O37.11 | loss definition, delta, target-space, comparability, architecture, schema and criterion tests |
| O37.12–O37.16 | common data, training config, initialization, sample order and dropout/RNG audits |
| O37.17 | optional isolated first-batch gradient probe |
| O37.18–O37.21 | optimizer budget, provenance, reused MSE and verified new Huber evidence |
| O37.22–O37.29 | metrics, loss effects, Huber regimes, gradients, clipping, optimization, convergence and runtime |
| O37.30 | optional generalization diagnostics |
| O37.31–O37.34 | hypothesis outcomes, findings, winner and Phase 38 reference update |
| O37.35 | figures S15_01–S15_08; S15_09 optional |
| O37.36–O37.41 | sweep tests, discrepancy log, summary, report, README and signed Phase 37 sign-off |

Canonical root: `artifacts/sweeps/S15_loss/`. New run evidence stays under `artifacts/runs/<run_id>/`; checkpoints must not be duplicated into the sweep root.

O37.17, O37.30 and S15_09 may be omitted only with an explicit machine-readable and human-readable reason. No runtime metric, winner, figure, report conclusion or sign-off may be fabricated before Huber completes and BEST is verified.

## Planned implementation ownership after Human approval

1. Create `src/course_work/sweeps/loss.py` as the Phase 37 contract/preflight owner.
2. Extend `src/course_work/experiments/phase_execution.py` with S15 dependency and condition specifications.
3. Extend `src/course_work/experiments/sweep_recovery.py` only if S15 read-only recovery/audit support is directly required.
4. Extend `src/course_work/sweeps/sweep_results.py` with S15 result verification, diagnostic/output schemas, full-precision winner rule, warning propagation and idempotent Phase 38 handoff/sign-off materialization.
5. Extend `src/course_work/reporting/phase_summary.py` with an allowlisted presentation payload and JSON processing-log route.
6. Extend `scripts/run_single_condition.py` with a guarded `S15_LOSS` Huber-only fresh-run path and MSE reuse-only enforcement.
7. Extend `scripts/run_all_pending.py` with Phase 37 audit/dry-run support while keeping automatic scientific execution blocked.
8. Add focused Phase 37 unit/integration/contract tests and only update directly affected shared tests.
9. Update `docs/RULE_BASE/architecture_rule.md` to register the approved Phase 37 owner and boundaries.
10. Add a short presentation/orchestration-only Phase 37 notebook section that reads processing JSON and canonical artifacts without training or Test access.

No implementation file listed above may be edited before Human approval of this plan.

## File and log routing

- reusable logic: `src/course_work/`
- scripts: `scripts/`
- tests: `tests/`
- scientific artifacts: `artifacts/`
- raw terminal logs: `artifacts/sweeps/logs/*.log`
- machine-readable process logs: `docs/save_log_in_processing/*.json`
- notebook: `notebook_course_work/CourseWork.ipynb`, presentation/orchestration only

Raw `.log` files must never be placed in `docs/save_log_in_processing/`. Processing JSON is not canonical scientific evidence.

## Phase 38 handoff contract

Preparation must define but not materialize a scientific handoff that preserves the selected Phase 37 loss exactly and changes only `max_epochs`:

- E50: reuse the exact Phase 37 winner if the frozen contract matches;
- E100: one fresh run;
- early stopping remains enabled;
- if Huber wins, `delta=1.0` remains fixed in model-space;
- Test remains forbidden.

Phase 38 execution and output creation are outside this plan.

## Stop and rollback conditions

Stop before Huber authorization if any of the following occurs:

- Phase 36 checksum, status, approval, winner or reference disagreement;
- unresolved critical inherited warning;
- MSE reference frozen-contract or BEST-provenance mismatch;
- missing/invalid signed YS1 scaler or unresolved raw-Wh equivalent delta;
- any architecture, parameter-count/schema, output-shape or optimizer-coverage difference;
- criterion class, reduction, delta, target-space or broadcasting mismatch;
- population, scaler, loader, optimizer, budget, seed, metric or Training Engine mismatch;
- initialization/sample-order evidence is falsely asserted instead of marked `NOT_VERIFIABLE`;
- Experiment Registry cannot register exactly one fresh Huber run safely;
- Test loader, targets, predictions or metrics are accessed;
- scientific outputs would need fabricated values;
- a source change beyond the approved Phase-37-owned scope becomes necessary.

Before official Huber execution, preparation may be rolled back by reverting only Phase-37-owned implementation changes created under the later approval. It must never alter Phase 0–36 evidence or reset unrelated user changes.

## Terminal execution boundary

Preparation ends after focused tests, audit-only, dry-run, preflight and an optional guarded no-train smoke all pass. It must not call `TrainingEngine.train()` for the official Huber condition, register a scientific winner, finalize Phase 37, access Test, or execute Phase 38.

The exact manual Huber command will be emitted only after approved implementation proves all gates pass. Human must execute that command separately. Raw terminal output must be routed to a descriptive `.log` under `artifacts/sweeps/logs/` including Phase, condition and real run ID when available.

## Human approval gate

Required next approval:

```text
CW-PHASE37-S15-PREPARATION-001
```

Approval authorizes only Phase 37 preparation implementation, focused tests and non-training preflight checks described above. It does not authorize Huber scientific training, MSE retraining, Test access, winner finalization or Phase 38.
