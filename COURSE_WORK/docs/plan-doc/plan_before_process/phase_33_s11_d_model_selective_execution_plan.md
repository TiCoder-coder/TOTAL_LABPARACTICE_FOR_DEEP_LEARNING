# Phase 33 S11 d_model Selective Execution Plan

## 1. Plan identity

```text
Plan ID: CW-PHASE-33-S11-DMODEL-SELECTIVE-EXECUTION-v1
Phase ID: 33
Phase name: S11 d_model Sweep
Phase detail: Phase_33_S11_d_model_sweep.md
Current status: COMPLETED
Implementation readiness: IMPLEMENTED_AND_VERIFIED
Scientific execution readiness: COMPLETED_VALIDATION_ONLY
```

## 2. Objective

Implement Phase 33 as a terminal-owned selective controlled sweep comparing `D32` and `D64` without rerunning the notebook and without retraining a complete exact-match reference.

The notebook remains presentation-only. It may only call a public reporting API that loads the derived Phase 33 processing log and renders persistent static HTML.

Processing logs under `docs/save_log_in_processing` may restore presentation state. They must not be used as scientific evidence, as a winner source or as authorization to execute Phase 33.

## 3. Current project state

```text
Environment revision: valid
Current device: MPS
Phase 20 canonical evidence: invalid and incomplete
Phase 21 canonical evidence: invalid and incomplete
Phase 22-30: blocked by upstream canonical evidence
Phase 30 derived processing log: BLOCKED
Phase 31 derived processing log: BLOCKED
Phase 32 implementation: absent
Phase 32 canonical artifacts: absent
Phase 32 processing log: absent
Phase 33 implementation: absent
Phase 33 canonical artifacts: absent
Phase 33 processing log: absent
```

Phase 33 cannot perform scientific execution until every upstream gate through Phase 32 is canonical and checksum-valid.

## 4. Upstream dependencies

Phase 33 requires:

```text
Phase 20 and Phase 21 complete learned-baseline evidence
Phase 22 valid learning diagnostics
Phase 23 through Phase 31 valid sequential sweep handoffs
Phase 32 PASS or PASS_WITH_WARNING without unresolved critical issues
s10_dropout_winner.json
s10_reference_update.json
phase_32_signoff.json
approved_for_phase33 = true
valid current environment revision
Test firewall confirmed
```

The authoritative upstream sources are approved Phase details, the experiment registry, run configs, run artifacts, checkpoints, histories, checksums, sweep manifests, winner artifacts, reference updates and phase sign-offs.

Derived JSON processing logs are lower-authority presentation records only.

## 5. Downstream dependencies

Phase 34 may consume Phase 33 only after Phase 33 produces:

```text
s11_d_model_winner.json
s11_reference_update.json
phase_33_signoff.json
approved_for_phase34 = true
```

No Phase 34 execution is included in this plan.

## 6. Files that must be read before implementation

```text
working_rule.md
docs/RULE_BASE/architecture_rule.md
docs/RULE_BASE/rule_code.md
docs/plan-doc/plan_detail_for_each_phase/Phase_20_LSTM_baseline_run.md
docs/plan-doc/plan_detail_for_each_phase/Phase_21_Transformer_B0_run.md
docs/plan-doc/plan_detail_for_each_phase/Phase_22_Learning-curve_diagnostics.md
docs/plan-doc/plan_detail_for_each_phase/Phase_23_S1_Feature-set_sweep.md through Phase_32_S10_Dropout_sweep.md
docs/plan-doc/plan_detail_for_each_phase/Phase_33_S11_d_model_sweep.md
src/course_work/models/transformer_regressor.py
src/course_work/models/transformer_encoder_layer.py
src/course_work/models/positional_encoding.py
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
relevant unit, contract and integration tests
current canonical artifacts and derived Phase 30-32 logs
```

## 7. Planned ownership

### 7.1 `src/course_work/sweeps/d_model.py`

```text
S11 identity and candidate registry
Phase 32 canonical handoff validation
D32 and D64 frozen contract
D64 exact-reference reuse eligibility
D32 fresh-run preparation
d_model geometry and divisibility validation
architecture-role and allowed-shape-delta validation
parameter-count and capacity diagnostics
Test firewall
```

It must not own generic training, registry persistence, HTML rendering or notebook orchestration.

### 7.2 `src/course_work/experiments/phase_execution.py`

```text
register Phase 33 dependency and condition definitions
classify canonical evidence
select the smallest safe action
block when Phase 32 is invalid
dispatch only a missing authorized condition
```

### 7.3 `src/course_work/sweeps/sweep_results.py`

```text
validate complete D32 and D64 evidence
select the winner using full-precision Validation RMSE
apply D32 only on exact RMSE tie
materialize canonical winner, reference update and sign-off
```

### 7.4 `src/course_work/reporting/phase_summary.py`

```text
generate the derived Phase 33 processing log
render compact persistent static HTML
show prerequisite state, D32 and D64 evidence, winner or exact block reason
exclude raw JSON, checksums and technical dumps from notebook presentation
```

### 7.5 Terminal scripts

```text
support target Phase 33
run audit and dry-run without scientific writes
execute only the missing authorized Phase 33 condition
never invoke notebook execution
never materialize an upstream Phase merely to inspect it
```

### 7.6 Notebook

After Phase 32 has a canonical preceding notebook section, the only planned Phase 33 code cell is:

```python
from course_work.reporting.phase_summary import render_phase_resume
render_phase_resume(33)
```

No training, CSV loading, JSON decision logic, model construction, metric computation or artifact writing may enter the notebook.

## 8. Phase 33 scientific contract

```text
Sweep ID: S11_DMODEL
Version: SWEEP_S11_DMODEL-v1
D32: d_model = 32
D64: d_model = 64
Reference: D64 from the exact Phase 32 winner configuration
New run: D32 with fresh seed-42 objects
Frozen heads: H4
Frozen layers: N2
Frozen FFN width: F128
Primary metric: best Validation RMSE Wh
Selection direction: minimum
Exact tie rule: D32
Test access: forbidden
```

Expected inherent changes:

```text
D32 head dimension: 8
D64 head dimension: 16
D32 FFN ratio: 4
D64 FFN ratio: 2
parameter counts differ
D-dependent tensor shapes differ
```

These differences must be documented and must not be compensated by changing heads, layers, FFN width, learning rate, weight decay, dropout or batch size.

## 9. Phase requirement mapping

| Phase 33 requirement | Planned owner | Verification | Evidence |
|---|---|---|---|
| Phase 32 gate | `d_model.py` and `phase_execution.py` | reject missing or invalid winner, reference and sign-off | preflight audit |
| D32 and D64 only | `d_model.py` | reject unregistered width | run matrix and unit tests |
| H4, N2 and F128 frozen | `d_model.py` | geometry assertions | definition, MHA and FFN audits |
| Valid divisibility | model preflight | verify D32 and D64 are divisible by H4 | definition audit |
| Same population | execution and registry layers | target IDs and fingerprints equal | common-data audit |
| D64 exact reuse | `d_model.py` | exact config, lineage, checkpoint and history validation | provenance and preflight audit |
| D32 fresh run | terminal dispatch and Training Engine | fresh model, optimizer and loader objects | registry run and run artifacts |
| No warm start | `d_model.py` and tests | wrong-width strict load fails | unit-test artifact |
| Validation-only metrics | shared metrics and finalizer | no Test records | metrics and Test-firewall audit |
| Full-precision winner | `sweep_results.py` | minimum RMSE, D32 on exact tie | winner artifact |
| Capacity context only | `d_model.py` | no efficiency override outside exact tie | diagnostics and tests |
| Phase 34 handoff | `sweep_results.py` | selected width and fingerprints reload correctly | reference update and sign-off |
| Notebook presentation only | `phase_summary.py` and notebook | one public call, static HTML, no widget MIME | notebook-boundary test |

## 10. Planned files to change after approval and upstream readiness

```text
docs/RULE_BASE/architecture_rule.md
docs/save_log_in_processing/phase_33_s11_d_model_log.json
notebook_course_work/CourseWork.ipynb
src/course_work/sweeps/d_model.py
src/course_work/sweeps/sweep_results.py
src/course_work/experiments/phase_execution.py
src/course_work/experiments/sweep_recovery.py
src/course_work/reporting/phase_summary.py
scripts/run_single_condition.py
scripts/run_all_pending.py
scripts/run_phase_background.py
scripts/run_phase_background/_notebook.py
tests/contracts/test_selective_execution_policy.py
tests/unit/test_d_model.py
tests/unit/test_phase_execution.py
tests/unit/test_sweep_finalization.py
tests/unit/test_sweep_recovery.py
tests/unit/test_selective_phase_reporting.py
tests/integration/test_notebook_boundary.py
```

The exact file set must be reduced if existing public APIs already satisfy a requirement.

The approved implementation reduced the file set. `CourseWork.ipynb`, `scripts/run_phase_background/_notebook.py`, `src/course_work/experiments/sweep_recovery.py` and `tests/integration/test_notebook_boundary.py` were not changed for Phase 33 because the existing public boundaries were sufficient and notebook mutation was not authorized for this execution.

## 11. Files that must not change

```text
data/raw_data/**
data/interim/**
data/data_after_split/**
artifacts/contracts/**
historical environment revisions
valid upstream run artifacts
valid upstream checkpoints
Phase 0-32 signed evidence
stored notebook outputs outside approved Phase 33 cells
Test predictions and Test metrics
```

No existing output may be deleted or globally cleared.

## 12. Sequential implementation plan

### Step 1. Re-audit upstream canonical state

Inspect Phase 20 through Phase 32 using canonical evidence and checksums.

Verification:

```text
environment valid
earliest invalid Phase identified
processing logs not treated as completion evidence
no scientific writes
```

Stop if Phase 32 is not valid. Phase 33 must not bypass an incomplete Phase 32.

### Step 2. Amend architecture scope

After Human approval and only when Phase 32 ownership is canonical, extend architecture scope and Phase-to-module mapping through Phase 33.

Verification:

```text
no owner conflict
no circular dependency
notebook boundary unchanged
Phase 32 remains immediately before Phase 33
```

### Step 3. Capture preservation baselines

Record checksums and presence state for the notebook, Phase 32 artifacts and log, experiment registry, run roots and current sweep artifacts.

Verification:

```text
read-only baseline
cell IDs and stored outputs recorded
no global notebook output mutation
```

### Step 4. Implement the Phase 33 source owner

Implement Phase 32 handoff validation, D32/D64 contract resolution, geometry audits, reference-reuse gate, fresh-run preparation and Test firewall in `d_model.py`.

Verify focused unit tests before any terminal integration change.

### Step 5. Register Phase 33 selective state

Extend selective inspection with exact Phase 33 dependencies, conditions and canonical paths.

Verification:

```text
invalid Phase 32 resolves to UPSTREAM_INVALID and BLOCK
valid complete Phase 33 resolves to VALID_REUSABLE and RENDER_ONLY
missing D32 with valid D64 resolves to EXECUTE_MISSING_ONLY for D32
unsupported conditions fail
```

### Step 6. Extend terminal dispatch

Add Phase 33 audit, dry-run and missing-only execution support.

Verification:

```text
audit-only writes nothing
dry-run writes nothing
no notebook process is started
no upstream materializer is called
only D32 is scheduled when exact D64 evidence is reusable
```

### Step 7. Extend canonical finalization

Create Phase 33 finalization only from complete verified registry and run evidence.

Verification:

```text
exactly D32 and D64 present
only BEST checkpoint metrics ranked
full-precision Validation RMSE used
D32 chosen only on exact tie
no Test records
winner and reference checksums reload
```

### Step 8. Add derived log and HTML

Generate:

```text
docs/save_log_in_processing/phase_33_s11_d_model_log.json
```

Verify static widget-free HTML, compact aligned tables and exact upstream block reasons.

### Step 9. Add notebook presentation

This step is forbidden until the Phase 32 notebook section exists directly before Phase 33.

Append only the Phase 33 markdown section and one public reporting call.

Verification:

```text
notebook JSON valid
Phase 32 precedes Phase 33
all unrelated cell sources unchanged
all unrelated stored outputs unchanged
no processing logic in Phase 33 cells
```

### Step 10. Execute Phase 33 from terminal

This step is permitted only after every upstream Phase through Phase 32 passes canonical verification.

Expected execution:

```text
reuse exact D64 reference
run only fresh D32
verify D32 immediately
finalize Phase 33
refresh only the Phase 33 derived log and HTML payload
```

### Step 11. Regression verification

Run focused tests followed by related integration and contract tests.

Verify Phase 32 handoff immutability, registry immutability, the Test firewall, the notebook boundary and a machine-readable Phase 34 handoff without executing Phase 34.

### Step 12. Write execution report

Record changed files, commands, validation results, scientific execution state, limitations and residual blockers under `docs/plan-doc/plan_to_refactor&fix/`.

## 13. Expected outputs

Canonical outputs follow Section 128 of the Phase 33 detail under:

```text
artifacts/sweeps/S11_d_model/
```

The mandatory final handoff outputs are:

```text
s11_d_model_winner.json
s11_reference_update.json
phase_33_signoff.json
```

The new D32 checkpoint and history remain under `artifacts/runs/<run_id>/` and must not be duplicated.

## 14. Validation and regression strategy

```text
syntax and import checks
Phase 33 contract unit tests
geometry and divisibility tests
wrong-width checkpoint-load rejection
canonical evidence and checksum tests
selective-action tests
terminal audit and dry-run tests
Test-firewall contract tests
static HTML tests
notebook-boundary and output-preservation tests
Phase 32 to Phase 33 handoff regression tests
```

Every implementation step must be verified before the next step starts.

## 15. Known risks

```text
Phase 20 and Phase 21 evidence is currently incomplete.
Phase 30 and Phase 31 derived logs are blocked.
Phase 32 has no implementation or canonical evidence.
The current Phase 30 PASS label cannot be trusted without complete declared outputs.
Processing logs can be stale after environment recovery.
D64 reference reuse may fail exact-config validation.
D32 and D64 parameter shapes differ by design.
Runtime and memory measurements may be incomparable if historical provenance is incomplete.
Notebook insertion before Phase 32 exists would violate phase ordering.
```

## 16. Stop conditions

```text
Phase 32 is not canonical and approved for Phase 33.
Any upstream winner, reference, sign-off, checkpoint or history checksum is missing or invalid.
D64 is not an exact Phase 32 reference match.
The selected configuration requires hard-coded fallback values.
The implementation would require changing more than d_model.
Test access is detected.
The notebook would contain processing logic.
An unrelated stored notebook output would change.
An existing valid signed artifact would be overwritten.
An implementation step fails its immediate verification.
```

## 17. Rollback strategy

```text
Do not delete historical evidence.
Do not overwrite signed upstream artifacts.
Use revisioned correction for signed artifacts.
Restore only Phase 33 files from the preservation baseline if a pre-sign-off step fails.
Keep failed scientific runs in registry history with explicit failure status.
Do not rank or reuse failed or incomplete runs.
```

## 18. Acceptance criteria

Implementation acceptance requires:

```text
approved architecture ownership through Phase 33
Phase 32 canonical handoff validation
exact D32 and D64 registry
fresh D32 and exact reused D64 behavior
H4, N2 and F128 frozen
valid D-dependent geometry
complete Validation-only metrics
correct full-precision winner rule
complete canonical artifacts and checksums
Phase 34 reference update
Test untouched
static HTML presentation
notebook presentation-only boundary
all relevant tests pass
```

Scientific completion additionally requires all upstream canonical gates through Phase 32 to pass.

## 19. Approval gate

```text
Plan created: true
Source code changed: false
Notebook changed: false
Scientific execution started: false
Waiting for Human verification and approval: true
```

Approval of this plan authorizes only the planned Phase 33 implementation. It does not authorize bypassing or fabricating Phase 20-32 evidence, and it does not implicitly authorize implementing Phase 32.
