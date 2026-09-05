# Phase 34 Current-State Rebaseline Pre-Process Plan

## Plan identity

```text
Plan ID: CW-PHASE-34-CURRENT-STATE-REBASELINE-v1
Phase ID: 34
Phase name: S12 Head Sweep
Plan status: COMPLETED_WITH_WARNING
Issue reference: CW-PHASE-34-REBASELINE-001
Execution started: true
Scientific execution started: true
Test accessed: false
```

## Post-training completion update

The Human-authorized H2 run `RUN_TR_S12_0020_DE823D66` completed 22 epochs and selected epoch 12 with full-precision Validation RMSE `58.65958891509437 Wh`. The approved H4 historical reference `RUN_TR_S09_0016_AE0FB819` retained the lower Validation RMSE `58.08190056355405 Wh` and is the Phase 34 winner.

Phase 34 was finalized through the canonical finalizer with status `PASS_WITH_WARNING`. The selected handoff is `num_heads=4`, `head_dim=16`, `d_model=64`; warning `H4_SOURCE_ARTIFACT_RETENTION_INCOMPLETE` is preserved and `approved_for_phase35=true`. Test remained forbidden and Phase 35 was not executed.

## Objective

Prepare Phase 34 from the current repository state without retraining H4, running H2, accessing Test or changing the S12 scientific contract.

## Scientific contract

```text
H2 = 2 heads, TRAIN_NEW
H4 = 4 heads, REUSE_REFERENCE
Swept field = model.num_heads only
Selected d_model = Phase 33 winner
num_layers = 2
ffn_dim = 128
Primary metric = full-precision Validation RMSE Wh
Exact RMSE tie = H2
Test = FORBIDDEN
```

## Current state

```text
Phase 33 selection JSON: checksum-valid
Selected d_model: 64
H4 run ID: RUN_TR_S09_0016_AE0FB819
H4 Validation RMSE: 58.08190056355405 Wh
H4 checkpoint: missing
H4 training history: missing
H4 training log: missing
H4 Validation predictions: missing
H2 run: absent
Phase 34 canonical artifacts: absent
Phase 34 processing log: deleted and stale
Signed Python 3.10.11 executable: unavailable
Current .venv: Python 3.11.14, package not installed
Preflight: blocked
```

## Context-read manifest

| Order | File or scope | Relation | Status |
|---:|---|---|---|
| 1 | `working_rule.md` relevant workflow, plan, safety and code-discipline sections | Governance | READ |
| 2 | `docs/RULE_BASE/architecture_rule.md` Phase 34 ownership, artifact, notebook and test boundaries | Architecture | READ |
| 3 | `docs/RULE_BASE/rule_code.md` approval, sequential, recovery and validation gates | Execution governance | READ |
| 4 | `docs/plan-doc/plan_detail_for_each_phase/Phase_34_S12_Head_sweep.md` | Acceptance baseline | READ COMPLETE |
| 5 | Former committed Phase 34 plan and processing baselines | Prior-state evidence | READ FROM GIT |
| 6 | Phase 33 winner, reference and sign-off | Direct upstream handoff | READ AND CHECKSUM VERIFIED |
| 7 | `src/course_work/sweeps/heads.py` | Phase 34 scientific owner | READ |
| 8 | `src/course_work/experiments/phase_execution.py` | State and action owner | READ |
| 9 | `src/course_work/experiments/sweep_recovery.py` | Recovery audit owner | READ |
| 10 | `src/course_work/sweeps/sweep_results.py` | Finalization owner | READ |
| 11 | `src/course_work/reporting/phase_summary.py` Phase 34 paths | Processing-log and presentation owner | READ |
| 12 | `scripts/run_single_condition.py` | H2 terminal dispatch | READ |
| 13 | `scripts/run_all_pending.py` | Selective orchestration | READ |
| 14 | Direct Phase 34 tests | Verification | READ |
| 15 | `CourseWork.ipynb` Phase 34 cells | Presentation boundary | READ |

```text
WORKING_RULE_READ=true
ARCHITECTURE_RULE_READ=true
RULE_CODE_READ=true
PHASE_DETAIL_READ_COMPLETE=true
RELATED_FILE_READ_COMPLETE=true
UPSTREAM_REQUIRED_PHASES_HEALTHY=PASS_WITH_APPROVED_HISTORICAL_RETENTION_WARNING
USER_APPROVED_PREPROCESS_PLAN=true
NO_OTHER_ACTIVE_EXECUTION_TASK=true
```

## Direct dependencies

Required upstream evidence:

```text
artifacts/sweeps/S11_d_model/s11_d_model_winner.json
artifacts/sweeps/S11_d_model/s11_reference_update.json
artifacts/sweeps/S11_d_model/phase_33_signoff.json
artifacts/runs/RUN_TR_S09_0016_AE0FB819/config.json
artifacts/runs/RUN_TR_S09_0016_AE0FB819/status.json
artifacts/runs/RUN_TR_S09_0016_AE0FB819/checkpoints/best_checkpoint.pt
artifacts/runs/RUN_TR_S09_0016_AE0FB819/training_history.csv
artifacts/runs/RUN_TR_S09_0016_AE0FB819/training.log
artifacts/runs/RUN_TR_S09_0016_AE0FB819/metrics/best_validation_metrics.json
artifacts/runs/RUN_TR_S09_0016_AE0FB819/predictions/best_validation_predictions.csv
artifacts/experiments/experiment_registry.jsonl
```

Required runtime:

```text
Python 3.10.11 matching the approved environment contract
course-work installed from COURSE_WORK through the approved package workflow
MPS or CUDA available as declared by the environment contract
```

## Downstream dependencies

Phase 35 remains blocked until Phase 34 produces checksum-valid:

```text
artifacts/sweeps/S12_heads/s12_head_winner.json
artifacts/sweeps/S12_heads/s12_reference_update.json
artifacts/sweeps/S12_heads/phase_34_signoff.json
approved_for_phase35 = true
```

## Files allowed to change after separate Human approval

The exact list must be reduced to the smallest proven set during implementation:

```text
src/course_work/sweeps/heads.py
src/course_work/experiments/phase_execution.py
src/course_work/experiments/sweep_recovery.py
src/course_work/sweeps/sweep_results.py
src/course_work/reporting/phase_summary.py
scripts/run_single_condition.py
scripts/run_all_pending.py
tests/unit/test_heads.py
tests/unit/test_condition_runner_config.py
tests/unit/test_phase_execution.py
tests/unit/test_sweep_finalization.py
tests/unit/test_sweep_recovery.py
tests/unit/test_selective_phase_reporting.py
tests/contracts/test_selective_execution_policy.py
```

The notebook will remain unchanged unless a later focused notebook-boundary failure proves a Phase 34 presentation correction is necessary and the overlapping dirty state can be preserved.

## Files forbidden to change

```text
Phase 33 winner, reference and sign-off
RUN_TR_S09_0016_AE0FB819 historical evidence
Experiment registry historical records
Training Engine scientific behavior
Data split, scaling, window population and metric contracts
Any Test artifact or Test authorization state
Any Phase outside the direct Phase 33 to Phase 34 chain
Unrelated dirty notebook content
```

## Required precondition recovery

### Gate R1: apply the approved H4 historical-reference protocol

The four missing original files are unrecoverable. Preserve their expected paths and checksums as historical retention metadata:

```text
best_checkpoint.pt = 3ccf735488336340275a4dccf040fcd17b98a4fa4746d3e665cf14f97428d75d
training_history.csv = 635e00dfcd032c3664c5e031b13f38d94caa3f73c4c6bf142eb0d6883d9eb43b
training.log = f709354699e8fb103b226fbae64ac2d2611fd80854ae1342246e86c94415d5d7
best_validation_predictions.csv = 9bb212edb469f8fac02cf9179a401fec04d27487131a1e17a5e14179b94a5a4e
```

Required acceptance mode:

```text
HISTORICAL_REFERENCE_WITH_INCOMPLETE_ARTIFACT_RETENTION
PASS_WITH_WARNING
Do not retrain H4
Do not fabricate missing files
Do not alter the historical registry record
```

Cross-validate the retained Phase 33 winner, reference update, sign-off, Experiment Registry record, H4 config, H4 status and H4 Validation metrics. Any conflicting retained value remains a hard block.

### Gate R2: restore approved runtime

Provision a Python 3.10.11 environment consistent with `pyproject.toml`, install the package canonically and verify the signed-environment recovery policy. Do not use notebook `sys.path` injection or a permanent `PYTHONPATH` workaround.

Failure action:

```text
STOP
Do not run focused tests under an unapproved scientific runtime and report them as authoritative.
```

## Planned corrective implementation

These steps may begin under the Human-approved corrective protocol. Scientific H2 execution remains blocked until Gate R2 and every Phase 34 preflight check pass.

### Step 1: re-run read-only Phase 34 state inspection

Verify:

```text
Phase 33 handoff valid
H4 exact reference valid
H2 missing
Phase 34 state CONDITION_INCOMPLETE
resolved action EXECUTE_MISSING_ONLY
effective action BLOCK without scientific authorization
Test FORBIDDEN
```

### Step 2: strengthen the Phase 34 H4 reuse gate if required

Ensure every Phase 34 authorization path requires the retained H4 config, status, metrics and registry evidence to agree, while carrying the approved missing checkpoint, history, log and prediction paths as explicit incomplete-retention metadata. `inspect_phase_state`, `plan_phase_resume`, `prepare_phase_34_condition` and `run_all_pending` must agree on the same historical-reference mode.

Validation:

```text
missing H4 checkpoint/history/log/predictions produce PASS_WITH_WARNING
retained config/status/metrics/registry mismatch blocks H2
missing retained evidence blocks H2
checksum mismatch on retained evidence blocks H2
no H4 retraining path is selected
```

### Step 3: enforce the one-factor H2 configuration delta

Before H2 registry registration, compare the canonical H2 config with the exact H4 source config. Allowed scientific difference:

```text
model.num_heads: 4 -> 2
```

Derived evidence may include `head_dim: 16 -> 32`, but no frozen data, model, optimization, runtime-protocol or lineage field may drift silently.

Validation:

```text
d_model remains 64
num_layers remains 2
ffn_dim remains 128
all data and training fields remain equal
population fingerprint remains equal
Test authorization remains false
```

### Step 4: verify checkpoint configuration provenance

Ensure Phase 34 verification can prove from checkpoint metadata:

```text
d_model
num_heads
derived head_dim
num_layers
ffn_dim
dropout
activation
pooling
model config fingerprint
```

Do not infer head count from state-dict shapes alone.

### Step 5: complete Phase 34 artifact schemas

Map every required Phase-detail output to its required fields. Correct only the Phase 34 finalizer and focused tests needed to enforce:

```text
full-precision Validation metrics
H2 and H4 geometry
parameter schema and count equality
config delta
common data population
training configuration
initialization and sample-order status
dropout scope
attention API
optimizer budget
run provenance
winner and Phase 35 handoff
complete Phase 34 sign-off fields
Test FORBIDDEN
```

Do not create canonical Phase 34 outputs before both H2 and H4 are verified.

### Step 6: strengthen focused tests

Add or update only tests that prove the Phase 34 contract and the corrections above. Tests must verify schema fields, not only file existence.

### Step 7: run focused Phase 34 tests sequentially

Planned order:

```text
tests/unit/test_heads.py
tests/unit/test_condition_runner_config.py
tests/unit/test_phase_execution.py Phase 34 cases
tests/unit/test_sweep_recovery.py Phase 34 cases
tests/unit/test_sweep_finalization.py Phase 34 cases
tests/unit/test_selective_phase_reporting.py Phase 34 cases
tests/contracts/test_selective_execution_policy.py Phase 34 cases
```

Stop on the first unexplained failure. Do not run the full repository suite unless a cross-cutting shared contract is changed.

### Step 8: run audit-only and dry-run

Run Phase 34 through the canonical installed package and terminal scripts with no scientific authorization.

Required results:

```text
H4 PASS_WITH_WARNING HISTORICAL_REFERENCE_WITH_INCOMPLETE_ARTIFACT_RETENTION
H2 MISSING
resolved action EXECUTE_MISSING_ONLY
effective action BLOCK without authorization
audit-only writes no scientific artifact
dry-run writes no scientific artifact
Test remains forbidden
```

### Step 9: refresh derived Phase 34 processing log only through its owner

After the state is valid and the reporting path is tested, generate the derived processing log through `course_work.reporting.phase_summary`. Do not hand-edit a derived log to hide missing canonical evidence.

### Step 10: stop before H2

Return `READY_FOR_H2` only when all non-training gates pass. Provide the exact canonical terminal command but do not execute it.

## Validation strategy

```text
Static syntax validation for changed Python files
Focused Phase 34 unit and contract tests
Read-only Phase 33 handoff checksum audit
Read-only H4 registry and artifact audit
Phase 34 audit-only invocation
Phase 34 dry-run invocation
Before-and-after checksum comparison proving no scientific artifact write
Git diff review restricted to approved Phase 34 files
No-comment and no-icon audit for new code
Notebook Phase 34 boundary inspection if notebook remains in affected scope
```

## Regression strategy

No full repository regression is planned unless a shared interface is changed. If a shared selective-execution or finalization interface changes, run its directly affected Phase 31-34 tests only, then reassess whether full regression is required.

## Expected outputs before H2

```text
Phase 34 state inspection showing H4 verified and H2 missing
Focused test report
Audit-only report
Dry-run report
Current Phase 34 processing log derived from canonical evidence
Exact H2 terminal command
No Phase 34 scientific result artifact
No H2 run
No Test access
```

## Acceptance criteria

```text
H4 retained evidence cross-validation passes with the approved incomplete-retention warning.
Approved Python 3.10.11 runtime available.
course-work package imports canonically.
Phase 33 handoff passes.
H2 is the only TRAIN_NEW condition.
H4 is REUSE_REFERENCE and is not retrained.
Only num_heads changes scientifically.
Focused Phase 34 tests pass.
Audit-only and dry-run pass without scientific writes.
Test remains forbidden.
Phase 34 reports READY_FOR_H2 but H2 has not started.
```

## Stop conditions

```text
Any retained H4 evidence is missing, conflicting or checksum-invalid.
Signed runtime remains unavailable.
Phase 33 lineage mismatch appears.
Any non-head scientific config field differs.
Any Test access is detected.
Any audit-only or dry-run writes scientific artifacts.
Any unexplained focused test failure occurs.
Any change outside the approved Phase 34 file list becomes necessary.
Any overlap with unrelated dirty notebook content cannot be preserved.
```

## Rollback strategy

Each approved logical code change will be isolated and verified before the next step. Revert only files changed by that approved step using an explicit file list. Never use repository-wide reset, restore or clean commands. Preserve all user-owned dirty files and historical evidence.

## User approval gate

```text
PLAN_CREATED=true
PLAN_VERIFIED_AGAINST_CURRENT_REPOSITORY_STATE=true
USER_APPROVED_PREPROCESS_PLAN=true
UPSTREAM_H4_EVIDENCE_HEALTHY=PASS_WITH_WARNING
SIGNED_RUNTIME_HEALTHY=false
IMPLEMENTATION_AUTHORIZED=true
SCIENTIFIC_EXECUTION_AUTHORIZED=false
```

Phase 34 corrective code and focused tests are authorized. H2 command authorization remains forbidden until the official environment recovery gate and all non-training preflight checks pass.

## Corrective implementation checkpoint

```text
H4_MODE=HISTORICAL_REFERENCE_WITH_INCOMPLETE_ARTIFACT_RETENTION
H4_EVIDENCE_STATUS=PASS_WITH_WARNING
H4_RETAINED_EVIDENCE_CROSS_VALIDATION=PASS
H4_MISSING_ARTIFACT_COUNT=4
H4_RETRAINED=false
H2_CREATED=false
H2_TRAINED=false
TEST_ACCESSED=false
STATIC_COMPILATION=PASS
FOCUSED_RUNTIME_TESTS=BLOCKED_SIGNED_PYTHON_3_10_11_UNAVAILABLE
AUDIT_ONLY_ENTRYPOINT=BLOCKED_CANONICAL_PACKAGE_IMPORT_UNAVAILABLE
SCIENTIFIC_EXECUTION_AUTHORIZED=false
```
