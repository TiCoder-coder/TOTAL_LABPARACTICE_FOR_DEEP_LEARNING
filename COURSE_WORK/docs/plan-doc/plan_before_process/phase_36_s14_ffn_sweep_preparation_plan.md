# Phase 36 — S14 FFN Sweep Pre-Process Plan

## Approval state

- Phase ID: `36`
- Phase name: `S14 FFN Sweep`
- Plan ID: `CW-PHASE36-S14-PREPARATION-001`
- Execution status: `PREPARATION_IMPLEMENTED_WAITING_FOR_MANUAL_F64_EXECUTION`
- This document was approved for preparation implementation only. Scientific training, Test access and Phase 37 execution remain separately forbidden.

## Sources of truth and alignment

The preparation is governed by:

- `docs/plan-doc/plan_detail_for_each_phase/Phase_36_S14_FFN_sweep.md`
- `docs/plan-doc/plan_overview/Main_plan.md`
- `docs/RULE_BASE/rule_code.md`
- `docs/RULE_BASE/architecture_rule.md`
- `working_rule.md`
- the signed Phase 35 handoff under `artifacts/sweeps/S13_layers/`

The requested overview path `docs/plan-doc/plan_overview/plan.md` does not exist. The repository-owned overview is `Main_plan.md`. Its S14 declaration is `64`, `128`, and `256`, which agrees with the Phase Detail. No scientific conflict was found.

## Objective

Prepare the canonical Phase 36 infrastructure for a Validation-only controlled sweep of Transformer FFN width. The only controlled scientific factor is `model.ffn_dim`. The implementation must reuse the exact selected Phase 35 reference for F128, prepare fresh seed-42 F64 and F256 conditions, audit FFN-only architecture changes, and stop before scientific training.

## Scientific contract

- Sweep ID: `S14_FFN`
- Sweep version: `SWEEP_S14_FFN-v1`
- Controlled field: `model.ffn_dim`
- F64: `ffn_dim=64`, `TRAIN_NEW`
- F128: `ffn_dim=128`, `REUSE_REFERENCE`
- F256: `ffn_dim=256`, `TRAIN_NEW`
- Selection metric: full-precision Validation RMSE Wh
- Selection direction: minimum
- Exact tie: smallest FFN width, ordered `F64 < F128 < F256`
- Test access: `FORBIDDEN`
- Phase 37 execution: `FORBIDDEN`
- F128 retraining: `FORBIDDEN`
- F64/F256 training during preparation: `FORBIDDEN`

## Canonical Phase 35 upstream handoff

Required inputs:

- `artifacts/sweeps/S13_layers/s13_layer_winner.json`
- `artifacts/sweeps/S13_layers/s13_reference_update.json`
- `artifacts/sweeps/S13_layers/phase_35_signoff.json`
- `artifacts/experiments/experiment_registry.jsonl`
- `artifacts/runs/RUN_TR_S09_0016_AE0FB819/config.json`
- retained status and Validation metrics for `RUN_TR_S09_0016_AE0FB819`

Read-only validation with the required Python 3.10 interpreter reports the Phase 35 sign-off as valid with status `PASS_WITH_WARNING` and no checksum issue. The handoff explicitly approves Phase 36 and declares:

```text
F64  = TRAIN_NEW
F128 = REUSE_REFERENCE
F256 = TRAIN_NEW
```

The inherited warning `H4_SOURCE_ARTIFACT_RETENTION_INCOMPLETE` remains active. The F128 historical reference is not to be retrained and its unrecoverable checkpoint/history/prediction/log files must not be fabricated. Phase 36 must cross-validate its retained config, registry, status, metrics, Phase 33/34/35 lineage, and signed Phase 35 assertions, then propagate the warning into all applicable Phase 36 outputs and the Phase 37 handoff.

## Dynamically resolved frozen configuration

The implementation must load these values from the signed Phase 35 handoff and referenced run config rather than hard-code the selected winner:

| Field | Resolved value |
|---|---:|
| Reference run | `RUN_TR_S09_0016_AE0FB819` |
| Feature variant | `FS2_TF1` |
| Target scaling | `YS1` |
| Lookback | `L36` |
| Pooling | `LAST_STEP` |
| Activation | `GELU` |
| Batch | `B32` |
| Learning rate | `0.0003` |
| Weight decay | `0.001` |
| Dropout | `0.1` |
| `d_model` | `64` |
| `num_heads` | `4` |
| `head_dim` | `16` |
| `num_layers` | `2` |
| Current `ffn_dim` | `128` |
| Loss | `MSE` |
| Maximum epochs | `50` |
| Early-stopping patience | `10` |
| Gradient clipping | `1.0` |
| Seed | `42` |
| Population | `WINDOWPOP-v1` |
| Population fingerprint | `a40ded8802e90008535d268720bad9e9dcca5eee1436ddb359deea3ad39a1987` |
| Metric version | `METRICS-v1` |

Derived expansion ratios for the resolved `d_model=64` are:

```text
F64  = 1.0
F128 = 2.0
F256 = 4.0
```

Expansion ratio is derived context, not a controlled factor.

## Frozen-variable contract

Outside `model.ffn_dim`, keep identical:

- feature set, target scaling, split membership, Train and Validation sample IDs/order policy;
- input size, `d_model`, head count, head dimension, layer count, sinusoidal PE, POST_NORM policy, masks, LAST_STEP pooling and regression head;
- GELU activation, dropout probability and dropout sites;
- AdamW topology, learning rate, weight decay, MSE, epoch cap, early stopping, `min_delta`, gradient clipping, scheduler/warmup policy and accumulation;
- target transform, X scaler, Training Engine, metric version, device policy and seed.

Forbidden compensations include mixed per-layer widths, changes to D/H/N, gated FFN, an added FFN layer, added FFN normalization, FFN-specific LR/WD/dropout, LayerDrop/DropPath, deep supervision, layer averaging, repeated PE, warm-starting, weight slicing/padding/interpolation, optimizer-state reuse, score-based reruns, and hidden range extension to F32/F512.

## Current repository state

### Existing reusable infrastructure

- The contract registry already declares `F64`, `F128`, and `F256` as FFN condition IDs.
- The experiment family registry already declares Phase 36 family `S14_FFN` as planned.
- Phase 35 now emits the required Phase 36 condition policy and approved signed handoff.
- Transformer construction already accepts `ffn_dim`.
- Shared experiment registry, Training Engine, metric implementation, dataset/loaders, target scaler, artifact utilities and sweep finalization patterns exist.

### Missing Phase-36-owned implementation

- `src/course_work/sweeps/ffn.py` does not exist.
- `experiments/phase_execution.py` only supports Phase 23–35.
- `experiments/sweep_recovery.py` has no Phase 36-specific dependency contract.
- `sweeps/sweep_results.py` only finalizes through Phase 35 and has no S14 output schemas.
- `reporting/phase_summary.py` only renders selective Phase 23–35 state.
- `scripts/run_single_condition.py` has no `S14_FFN` dispatch or FFN-only delta guard.
- `scripts/run_all_pending.py` has no Phase 36 registration or two-condition manual-training guard.
- no focused `tests/unit/test_ffn.py` or Phase 36 integration/contract coverage exists.
- `CourseWork.ipynb` has no Phase 36 presentation section.
- `architecture_rule.md` still defines Phase 36+ as outside the implemented scope and therefore must be updated only after separate Human approval.
- no canonical `artifacts/sweeps/S14_ffn/` scientific outputs or Phase 36 processing JSON exist, as expected before execution.

Current implementation classification: `MISSING`.

## Preliminary read-only architecture evidence

A disposable model inspection using the exact Phase 35 reference configuration and required Python 3.10 interpreter produced:

| Condition | FFN width | Trainable parameters | Pairwise delta |
|---|---:|---:|---:|
| F64 | 64 | 52,673 | — |
| F128 | 128 | 69,185 | +16,512 from F64 |
| F256 | 256 | 102,209 | +33,024 from F128 |

The state-dict key sets were equal. Only `encoder.layers.*.linear1.{weight,bias}` and `encoder.layers.*.linear2.weight` changed shape; MHA and all other tensor shapes remained invariant. The observed deltas agree with the Phase Detail audit formula `N * (2D + 1) * delta_M` for `D=64`, `N=2`.

This evidence is diagnostic only. It is not a canonical Phase 36 artifact and does not replace the approved implementation, focused tests, preflight, or post-training finalization.

## Planned implementation ownership

After Human approval only:

1. Create `src/course_work/sweeps/ffn.py` as the Phase 36 contract owner.
   - load and validate the signed Phase 35 handoff;
   - resolve all inherited values dynamically;
   - define F64/F128/F256 and execution modes;
   - enforce the exact F128 reuse gate and inherited-warning policy;
   - audit FFN geometry, state-dict shape whitelist, parameter monotonicity/deltas, MHA and non-FFN invariance, optimizer coverage, attention API, initialization/sample-order policy and Test firewall;
   - prepare F64/F256 only after all gates pass.
2. Extend `src/course_work/experiments/phase_execution.py` with the canonical Phase 36 dependency and condition specification.
3. Extend `src/course_work/experiments/sweep_recovery.py` only where required for Phase 36 read-only recovery/audit inspection.
4. Extend `src/course_work/sweeps/sweep_results.py` with S14 Validation-only result validation, exact-tie ordering, all required/optional output schemas, warning propagation, winner materialization, Phase 37 handoff and idempotent sign-off.
5. Extend `src/course_work/reporting/phase_summary.py` with an allowlisted Phase 36 presentation payload and JSON processing-log route.
6. Extend `scripts/run_single_condition.py` with `S14_FFN`, F64/F256 fresh-training dispatch, F128 reuse-only behavior, and a strict `model.ffn_dim`-only config delta check.
7. Extend `scripts/run_all_pending.py` with Phase 36 audit/dry-run support while keeping automatic expensive execution blocked.
8. Add focused Phase 36 tests and update directly affected shared tests without weakening existing assertions.
9. Update `docs/RULE_BASE/architecture_rule.md` to declare the approved Phase 36 owner, boundaries, log routing and notebook responsibility.
10. Add a short presentation/orchestration-only Phase 36 section to `notebook_course_work/CourseWork.ipynb`. It may load the processing JSON and canonical outputs but may not train, access Test, or own scientific logic.

## Expected files changed after approval

Planned source and scripts:

- `src/course_work/sweeps/ffn.py` new
- `src/course_work/experiments/phase_execution.py`
- `src/course_work/experiments/sweep_recovery.py` only if direct Phase 36 support is required
- `src/course_work/sweeps/sweep_results.py`
- `src/course_work/reporting/phase_summary.py`
- `scripts/run_single_condition.py`
- `scripts/run_all_pending.py`

Planned tests:

- `tests/unit/test_ffn.py` new
- directly affected Phase execution, recovery, runner-config, selective-reporting and sweep-finalization tests

Planned docs and presentation:

- `docs/RULE_BASE/architecture_rule.md`
- this pre-process plan, with a post-implementation checkpoint appended only after approval
- `notebook_course_work/CourseWork.ipynb`

Runtime outputs after approved preparation checks:

- `docs/save_log_in_processing/phase_36_s14_ffn_log.json` only

Scientific outputs under `artifacts/sweeps/S14_ffn/` must not be fabricated during preparation. A preflight-only artifact may be written only if the existing architecture and approved implementation explicitly make it canonical and clearly non-scientific. Winner, metrics, training provenance, figures, summary, report, README, reference update and sign-off require completed scientific evidence.

## Files that must not change

- Phase 35 winner, reference update, sign-off and retained run evidence
- any Phase 0–35 scientific result or checkpoint
- raw/interim/processed data and signed scaler/window/metric artifacts
- experiment status or run record except a future approved F64/F256 run registration by the manual runner
- Test authorization, Test targets, Test metrics or Test predictions
- Phase 37 source, artifacts, notebook execution or approval state
- the Phase Detail and overview scientific contract

## Phase Detail requirement mapping

| Requirement group | Implementation owner | Verification | Evidence after scientific completion |
|---|---|---|---|
| O36.1–O36.4 manifest, contract, preflight, run matrix | `ffn.py`, `phase_execution.py`, `sweep_results.py` | contract/unit tests, audit-only, dry-run | S14 manifest/contract/preflight/run-matrix files |
| O36.5–O36.15 FFN, architecture, shape, parameter, invariance, optimizer and config audits | `ffn.py` | disposable F64/F128/F256 models; schema assertions | named S14 audit CSVs |
| O36.16 unit tests | focused Phase 36 tests | exact interpreter pytest | `s14_ffn_unit_tests.csv` after finalization |
| O36.17–O36.24 common data/training/init/order/dropout/attention/budget | `ffn.py`, runner and shared loaders/engine | lineage/config diff, optimizer and attention sanity | named S14 audit CSVs; optional shared-prefix audit |
| O36.25–O36.28 provenance and condition evidence | registry, runner, `sweep_results.py` | F128 reuse gate; strict fresh-run verification for F64/F256 | provenance plus reused/new run evidence |
| O36.29–O36.41 metrics through Phase 37 handoff | `sweep_results.py` | full-precision ranking and idempotent finalization tests | metrics, effects, diagnostics, winner and reference update |
| O36.42 figures | `sweep_results.py`/source-owned plotting | file/schema/checksum tests | S14_01 through S14_09; S14_10 optional |
| O36.43–O36.48 tests, discrepancy, summary, report, README, sign-off | `sweep_results.py` | output completeness/checksums/sign-off validation | exact named files under `S14_ffn/` |
| Warning propagation | `ffn.py`, finalizer, reporting | inherited-warning assertions | manifest, winner, summary, report, reference update, sign-off |
| Test firewall | all Phase 36 owners | source scan, request guards, registry assertions | `test_status=FORBIDDEN`; no Test evidence |
| Notebook boundary | `phase_summary.py`, notebook | cell audit and presentation-only execution | no training/Test code; canonical artifact display only |

## Implementation order after approval

1. Revalidate Phase 35 checksums, `approved_for_phase36`, F128 reference identity and Test firewall.
2. Add Phase 36 owner and pure/read-only architecture/preflight APIs.
3. Add selective execution, recovery, runner and reporting registrations.
4. Add focused tests before enabling any manual scientific command.
5. Add finalization schemas and exact full-precision tie behavior.
6. Add the notebook presentation section and architecture-rule update.
7. Run focused tests with the exact interpreter.
8. Run Phase 36 audit-only.
9. Run Phase 36 dry-run.
10. Build/read the Phase 36 preflight and confirm parameter monotonicity, FFN-only deltas, MHA/non-FFN invariance, optimizer coverage, F128 reuse, only F64/F256 `TRAIN_NEW`, and Test forbidden.
11. Run a guarded disposable forward/backward sanity check for F64 and F256 only. Do not register a run and do not call the Training Engine.
12. Stop before F64 training and return the exact manual F64 command.

## Focused validation commands after approval

Use exactly:

```bash
cd COURSE_WORK
MPLCONFIGDIR=/private/tmp/course-work-mpl-cache \
PYTHONPATH=src \
/Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10 \
  -m pytest \
  tests/unit/test_ffn.py \
  tests/unit/test_phase_execution.py \
  tests/unit/test_sweep_recovery.py \
  tests/unit/test_condition_runner_config.py \
  tests/unit/test_selective_phase_reporting.py \
  tests/unit/test_sweep_finalization.py -q
```

Audit-only and dry-run commands will use `scripts/run_all_pending.py --phase-id 36 --audit-only` and `--dry-run` only after the script supports Phase 36. Both must write no scientific artifacts and must report F64/F256 as pending fresh runs and F128 as reusable.

## Regression strategy

- Re-run directly affected Phase 35 handoff/finalization tests because Phase 36 consumes that schema.
- Re-run Phase 34/35 sign-off validation read-only to ensure no upstream mutation.
- Avoid a full repository regression unless a shared cross-cutting failure proves it necessary.
- Verify `git diff` only for approved files and preserve all pre-existing dirty user files.
- Confirm repeat audit/dry-run and finalization paths are deterministic and idempotent.

## Required output classification

- Required after complete scientific execution: O36.1–O36.19, O36.21–O36.36, O36.38–O36.48.
- Optional: O36.20 shared-prefix initialization audit, O36.37 generalization diagnostics, `S14_10_generalization_gap_optional.png`, and peak-memory fields.
- Not applicable before training: run metrics, BEST evidence, pairwise scientific effects, winner, Phase 37 reference, scientific figures and Phase 36 sign-off.
- Optional omissions must be recorded honestly in the discrepancy log and report.

## Acceptance criteria for preparation

Preparation is ready for manual F64 training only when:

- source/detail/overview alignment is `PASS`;
- signed Phase 35 handoff remains valid and approves Phase 36;
- F128 passes the exact historical-reference reuse gate without retraining or fabricated evidence;
- the frozen configuration is dynamically resolved;
- F64 and F256 are the only `TRAIN_NEW` conditions;
- parameter counts are strictly `F64 < F128 < F256`;
- runtime parameter deltas match the FFN-only architecture delta;
- state-dict key sets are equal and only FFN-width tensor shapes differ;
- MHA and every non-FFN role remain invariant;
- optimizer coverage contains every trainable parameter exactly once;
- disposable F64/F256 forward/backward and attention sanity checks pass;
- focused tests, audit-only, dry-run and preflight pass;
- Test remains forbidden and no Phase 36 training has executed;
- raw log and JSON log routing comply with the project rules.

## Risks

- F128 is an inherited historical reference with incomplete artifact retention. This is a non-critical warning only if all retained signed evidence remains mutually consistent.
- Shared files currently contain pre-existing user changes. Implementation must patch narrowly and must not overwrite or clean unrelated work.
- Phase 36 support extends multiple shared phase-range registries. A missed range boundary could make audit/report/finalization inconsistent.
- The Phase Detail requires many exact schemas. Generic Phase 23–35 finalization cannot be extended by merely changing a range constant.
- Width-dependent RNG consumption means whole-state or dropout-mask equality is not a fairness requirement.
- A boundary winner is not authorization to expand the sweep.

## Stop and rollback conditions

Stop without training if any of the following occurs:

- overview and Phase Detail scientifically conflict;
- Phase 35 signed handoff or F128 identity/checksum/config/metric provenance is invalid;
- inherited F128 warning cannot be represented without fabrication or retraining;
- a config difference outside `model.ffn_dim` appears;
- any layer uses a different FFN width within one candidate;
- parameter monotonicity or formula audit fails;
- state-dict keys differ or a non-FFN tensor shape changes;
- optimizer coverage has missing or duplicate trainable parameters;
- F64/F256 sanity fails or accesses Test;
- a required implementation change falls outside this approved file list;
- tests, audit-only, dry-run or preflight fail for an unexplained reason;
- any command would train F64/F256, retrain F128, access Test, or execute Phase 37.

Before approved scientific execution, rollback is limited to the exact Phase-36-owned changes introduced by the approved implementation. Never reset, restore, clean, or overwrite unrelated repository changes.

## Log routing

- Raw future terminal/training logs: `artifacts/sweeps/logs/phase_36_<condition>_<run_id>_terminal.log`
- Machine-readable Phase/process log: `docs/save_log_in_processing/phase_36_s14_ffn_log.json`
- Never put `.log` files under `docs/save_log_in_processing/`.

## Human approval gate

No implementation or execution may proceed until the Human explicitly verifies and approves plan `CW-PHASE36-S14-PREPARATION-001` in a later message. After approval, execute only the preparation steps above and stop before the first F64 training command.

## Approved preparation checkpoint — 2026-08-24

- Human approval received for `CW-PHASE36-S14-PREPARATION-001`.
- Phase 36 owner, selective execution, runner, reporting, notebook presentation and focused tests were implemented.
- Phase 35 handoff validated as `PASS_WITH_WARNING`; F128 remains `REUSE_REFERENCE` and was not retrained.
- F64 and F256 are the only `TRAIN_NEW` conditions.
- Parameter monotonicity, FFN-only shape delta, MHA/non-FFN invariance, optimizer coverage and disposable F64/F256 forward/backward attention sanity all passed.
- Focused test result: `87 passed`.
- Audit-only: PASS; no scientific artifacts written.
- Dry-run: PASS; no scientific artifacts written.
- Processing snapshot: `docs/save_log_in_processing/phase_36_s14_ffn_log.json`.
- Test remained forbidden. No training and no Phase 37 execution occurred.
- Preparation stopped before the first F64 scientific training command.

## Finalized scientific checkpoint — 2026-08-24

- F64 `RUN_TR_S14_0022_AA048302` and F256 `RUN_TR_S14_0023_A711A9B8` were completed manually through the approved terminal runner; neither run was repeated during finalization.
- F128 remained the exact Phase 35 historical reference `RUN_TR_S09_0016_AE0FB819` and was not retrained.
- F64 and F256 BEST checkpoints strict-loaded into fresh exact architectures and reproduced full ordered Validation METRICS-v1 within the registered numerical tolerance.
- Full-precision Validation RMSE values were F64 `58.39803077736952`, F128 `58.08190056355405`, and F256 `57.69679988114431` Wh.
- F256 was selected by the predeclared minimum-RMSE rule with `ffn_dim=256`; its upper-boundary status is documented and no F512 condition was introduced.
- Canonical O36 artifacts, figures, winner, Phase 37 reference update and `phase_36_signoff.json` were materialized under `artifacts/sweeps/S14_ffn/`.
- Phase 36 status is `PASS_WITH_WARNING` because `H4_SOURCE_ARTIFACT_RETENTION_INCOMPLETE` remains inherited through F128.
- Phase 37 handoff policy is MSE `REUSE_REFERENCE` and Huber `TRAIN_NEW`. Phase 37 was not executed.
- Test remained forbidden and was not accessed.
