# Phase 32 S10 Dropout Selective Execution Plan

## 1. Plan identity

```text
Plan ID: CW-PHASE-32-S10-DROPOUT-SELECTIVE-EXECUTION-v1
Related issue: CW-PHASE-33-UPSTREAM-PHASE-32-MISSING-001
Phase ID: 32
Phase name: S10 Dropout Sweep
Phase detail: Phase_32_S10_Dropout_sweep.md
Status: WAITING_FOR_HUMAN_APPROVAL
Implementation readiness: READY_AFTER_APPROVAL
Scientific execution readiness: BLOCKED_BY_PHASE_31
```

## 2. Objective

Implement Phase 32 as a source-owned and terminal-executed controlled sweep for `DR01`, `DR02` and `DR03`, while preserving the notebook as a presentation-only consumer of a derived JSON log and persistent static HTML.

The implementation must be available even while scientific execution is blocked, but it must fail before model, loader or optimizer construction whenever Phase 31 canonical evidence is invalid.

## 3. Current state

```text
Phase 31 selective state: UPSTREAM_INVALID
Phase 31 verified conditions: none
Phase 31 missing conditions: WD0, WD1, WD2
Phase 32 source owner: absent
Phase 32 selective registry: absent
Phase 32 terminal dispatch: absent
Phase 32 processing log: absent
Phase 32 notebook section: absent
Phase 32 scientific artifacts: absent
```

The current safe Phase 32 result after implementation is expected to be a clear `UPSTREAM_INVALID/BLOCK` presentation until Phase 31 becomes canonical.

## 4. Upstream dependencies

Phase 32 scientific execution requires:

```text
Phase 31 PASS or PASS_WITH_WARNING without unresolved critical issues
s9_weight_decay_winner.json
s9_reference_update.json
phase_31_signoff.json
approved_for_phase32 = true
valid environment revision with MPS or CUDA
complete experiment registry lineage
complete reference checkpoint and history
Test firewall confirmed
```

Processing logs may be used for presentation continuity only and cannot authorize execution or reference reuse.

## 5. Downstream dependencies

Phase 33 requires:

```text
s10_dropout_winner.json
s10_reference_update.json
phase_32_signoff.json
approved_for_phase33 = true
```

Phase 32 must not create these outputs until all three registered dropout conditions are complete and verified.

## 6. Planned ownership

### 6.1 `src/course_work/sweeps/dropout.py`

```text
Phase 32 identity and candidate registry
Phase 31 handoff validation
DR01 exact-reference reuse gate
DR02 and DR03 fresh-run preparation
dropout-site discovery and frozen-scope fingerprint
dropout train/eval semantics validation
constant runtime dropout validation
same-parameter-schema policy
Test firewall
```

The module must not own generic training, experiment registry persistence, HTML rendering or notebook orchestration.

### 6.2 `src/course_work/experiments/phase_execution.py`

```text
Phase 32 dependency registry
DR01, DR02 and DR03 condition registry
canonical evidence classification
smallest-safe-action resolution
missing-only dispatch authorization
```

### 6.3 `src/course_work/sweeps/sweep_results.py`

```text
complete-condition validation
verified Validation-only comparison
full-precision winner selection
lower-dropout exact-tie rule
winner, reference update and signed artifact finalization
```

### 6.4 `src/course_work/reporting/phase_summary.py`

```text
derived Phase 32 processing log
compact static HTML
prerequisite state
condition evidence
winner or complete block reasons
```

### 6.5 Terminal scripts

```text
Phase 32 audit and dry-run
missing-only DR02 and DR03 dispatch
DR01 reuse without retraining
no notebook process
no upstream materializer invocation
```

### 6.6 Notebook

The Phase 32 notebook code cell may contain only:

```python
from course_work.reporting.phase_summary import render_phase_resume
render_phase_resume(32)
```

## 7. Scientific contract

```text
Sweep ID: S10_DROPOUT
Version: SWEEP_S10_DROPOUT-v1
DR01: dropout = 0.1
DR02: dropout = 0.2
DR03: dropout = 0.3
Reference: DR01 from exact Phase 31 winner configuration
New runs: DR02 and DR03
Model capacity: D64, H4, N2, F128
Optimizer: AdamW
Primary metric: best Validation RMSE Wh
Selection direction: minimum
Exact tie rule: lower dropout
Test access: forbidden
```

Only the registered global dropout probability may change at existing controlled dropout sites. No input, head, pooling, per-layer, scheduled or stochastic-depth dropout may be added.

## 8. Phase requirement mapping

| Phase 32 requirement | Planned owner | Verification | Evidence |
|---|---|---|---|
| Phase 31 gate | `dropout.py` and `phase_execution.py` | reject invalid winner, reference or sign-off | preflight audit |
| DR01, DR02 and DR03 only | `dropout.py` | reject unsupported probability | run matrix and unit tests |
| Frozen dropout sites | `dropout.py` | site topology and fingerprint equal | site registry and scope manifest |
| Correct train/eval modes | model and Training Engine audits | Train stochastic, Validation deterministic, train restored | mode audits |
| DR01 exact reuse | `dropout.py` | exact config and complete evidence | reference provenance |
| DR02 and DR03 fresh runs | terminal dispatch | fresh model, optimizer and loaders | registry and run artifacts |
| Same population | execution and registry layers | IDs and fingerprints equal | common-data audit |
| Same parameter schema | `dropout.py` | names, shapes and counts equal | architecture audit |
| Full-precision winner | `sweep_results.py` | minimum RMSE and lower dropout on exact tie | winner artifact |
| Phase 33 handoff | `sweep_results.py` | selected dropout and lineage reload correctly | reference update and sign-off |
| No Test | all owners | no Test loader, predictions or metrics | Test-firewall tests |
| Notebook presentation only | reporting and notebook | one public call and static HTML | notebook-boundary test |

## 9. Files to read before implementation

```text
working_rule.md
docs/RULE_BASE/architecture_rule.md
docs/RULE_BASE/rule_code.md
docs/plan-doc/plan_detail_for_each_phase/Phase_31_S9_Weight-decay_sweep.md
docs/plan-doc/plan_detail_for_each_phase/Phase_32_S10_Dropout_sweep.md
docs/plan-doc/plan_detail_for_each_phase/Phase_33_S11_d_model_sweep.md
src/course_work/models/transformer_regressor.py
src/course_work/models/transformer_encoder_layer.py
src/course_work/training/engine.py
src/course_work/experiments/registry.py
src/course_work/experiments/phase_execution.py
src/course_work/experiments/sweep_recovery.py
src/course_work/sweeps/sweep_results.py
src/course_work/sweeps/weight_decay.py
src/course_work/reporting/phase_summary.py
scripts/run_single_condition.py
scripts/run_all_pending.py
scripts/run_phase_background.py
notebook_course_work/CourseWork.ipynb
relevant tests and current Phase 31 evidence
```

## 10. Planned files to change after approval

```text
docs/RULE_BASE/architecture_rule.md
docs/save_log_in_processing/phase_32_s10_dropout_log.json
notebook_course_work/CourseWork.ipynb
src/course_work/sweeps/dropout.py
src/course_work/sweeps/sweep_results.py
src/course_work/experiments/phase_execution.py
src/course_work/experiments/sweep_recovery.py
src/course_work/reporting/phase_summary.py
scripts/run_single_condition.py
scripts/run_all_pending.py
scripts/run_phase_background.py
scripts/run_phase_background/_notebook.py
tests/contracts/test_selective_execution_policy.py
tests/unit/test_dropout.py
tests/unit/test_phase_execution.py
tests/unit/test_sweep_finalization.py
tests/unit/test_sweep_recovery.py
tests/unit/test_selective_phase_reporting.py
tests/integration/test_notebook_boundary.py
```

The implementation must minimize this set when an existing public API already satisfies the requirement.

## 11. Files that must not change

```text
data/raw_data/**
data/interim/**
data/data_after_split/**
artifacts/contracts/**
historical environment revisions
valid upstream signed evidence
valid upstream checkpoints and histories
stored notebook outputs outside approved Phase 32 cells
Test predictions and metrics
```

## 12. Sequential implementation plan

### Step 1. Capture preservation baselines

Record current notebook checksum, cell IDs, stored outputs, registry state, Phase 31 evidence and sweep artifacts.

Verify that baseline capture is read-only.

### Step 2. Amend architecture scope through Phase 32

Register `sweeps/dropout.py` as Phase 32 owner and add the Phase 32 notebook boundary.

Verify no owner conflict, dependency inversion or notebook processing permission is introduced.

### Step 3. Implement Phase 32 preflight

Implement Phase 31 handoff validation, registered dropout mapping, frozen configuration resolution and Test firewall.

Immediate verification:

```text
current invalid Phase 31 blocks before model construction
missing winner/reference/sign-off reasons are complete
no hard-coded upstream winner values
```

### Step 4. Implement dropout-scope inspection

Inspect actual Transformer-v1 dropout sites and generate a stable semantic scope fingerprint.

Immediate verification:

```text
MHA attention-weight dropout represented when controlled globally
residual and FFN dropout represented
no input, head or pooling dropout introduced
same site topology across DR01, DR02 and DR03
```

### Step 5. Implement condition preparation

Prepare exact DR01 reuse and fresh DR02/DR03 execution configurations.

Immediate verification:

```text
DR01 cannot retrain
DR02 and DR03 cannot warm-start
optimizer state cannot transfer
all non-dropout fields remain frozen
```

### Step 6. Register Phase 32 selective state

Extend canonical state inspection and action resolution.

Immediate verification:

```text
invalid Phase 31 resolves to UPSTREAM_INVALID and BLOCK
valid DR01 plus missing DR02/DR03 resolves to EXECUTE_MISSING_ONLY
complete Phase 32 resolves to VALID_REUSABLE and RENDER_ONLY
```

### Step 7. Extend terminal dispatch

Add audit, dry-run and missing-only execution support for Phase 32.

Immediate verification:

```text
audit-only and dry-run write no scientific artifacts
no notebook execution
no upstream materializer call
only unresolved authorized conditions execute
```

### Step 8. Extend canonical finalization

Finalize only from complete verified DR01, DR02 and DR03 registry evidence.

Immediate verification:

```text
all three conditions exactly once
only BEST Validation metrics ranked
full-precision RMSE
lower dropout only on exact tie
no Test records
all output checksums reload
```

### Step 9. Add derived log and static HTML

Generate `phase_32_s10_dropout_log.json` and render compact persistent HTML.

Immediate verification:

```text
no widget MIME
no JavaScript dependency
processing log not treated as scientific completion
current upstream block reason visible
```

### Step 10. Add notebook presentation

Insert Phase 32 immediately after Phase 31 and before the aggregate log cell. Add only a markdown description and the public rendering call.

Immediate verification:

```text
notebook JSON valid
all previous cell sources and outputs unchanged
no processing logic
Phase 32 cell runs independently after restart
```

### Step 11. Run tests

Run syntax, focused unit, contract, integration, notebook-boundary and preservation tests.

Stop and create a corrective plan if any verification fails.

### Step 12. Scientific execution gate

If Phase 31 is still invalid, stop with an accurate `UPSTREAM_INVALID/BLOCK` result and do not train.

If Phase 31 later becomes canonical, reuse DR01 and run only missing DR02/DR03 from terminal, verifying each run immediately before finalization.

### Step 13. Resume Phase 33

Resume the approved Phase 33 plan only after Phase 32 winner, reference update and sign-off are complete and checksum-valid.

## 13. Expected outputs

Canonical Phase 32 outputs follow Section 201 of the Phase detail under:

```text
artifacts/sweeps/S10_dropout/
```

The mandatory Phase 33 handoff is:

```text
s10_dropout_winner.json
s10_reference_update.json
phase_32_signoff.json
```

DR02 and DR03 run artifacts remain under their registry-owned `artifacts/runs/<run_id>/` roots.

## 14. Validation and regression strategy

```text
dropout candidate validation
dropout-site topology and scope fingerprint tests
train/eval mode-transition tests
eval-repeatability tests
same-parameter-schema tests
reference reuse and no-retrain tests
fresh-run and no-warm-start tests
canonical checksum tests
winner and tie-rule tests
selective action tests
terminal audit and dry-run tests
Test-firewall tests
static HTML tests
notebook-boundary and output-preservation tests
Phase 31 and Phase 33 handoff regression tests
```

## 15. Known risks

```text
Phase 31 is currently invalid.
Phase 20 and Phase 21 canonical evidence remains incomplete.
Processing logs may be stale after environment recovery.
Dropout probability is config provenance and is not encoded in state_dict tensors.
Validation in train mode would invalidate selection.
Failure to restore train mode would disable dropout in later training epochs.
Historical DR01 initialization or sample-order metadata may be incomplete.
```

## 16. Stop conditions

```text
Phase 31 canonical handoff is invalid before scientific execution.
DR01 is not an exact Phase 31 winner match.
Any non-dropout factor changes.
Dropout-site topology changes between candidates.
Validation does not run in eval mode.
Training mode is not restored after Validation.
Warm-start or optimizer-state reuse is detected.
Test access is detected.
An unrelated notebook output changes.
An existing signed artifact would be overwritten.
Any immediate verification fails.
```

## 17. Rollback strategy

```text
preserve all historical evidence
never delete failed runs
mark incomplete runs explicitly
do not rank incomplete evidence
restore only Phase 32 implementation and presentation files from preservation baselines
use revisioned correction for signed artifacts
```

## 18. Acceptance criteria

Implementation acceptance requires:

```text
architecture ownership through Phase 32
source-owned Phase 32 preflight and condition preparation
DR01, DR02 and DR03 exact registry
complete dropout-scope and mode semantics validation
safe terminal audit, dry-run and missing-only behavior
static HTML presentation
notebook presentation-only boundary
all focused and regression tests pass
```

Scientific completion additionally requires Phase 31 canonical validity, two verified fresh runs, one verified reused reference, complete signed Phase 32 artifacts and no Test access.

## 19. Approval gate

```text
Corrective plan created: true
Phase 32 source code changed: false
Notebook changed: false
Scientific execution started: false
Waiting for Human verification and approval: true
```

Approval of the Phase 33 plan does not authorize Phase 32 implementation. This corrective plan requires separate explicit Human approval.
