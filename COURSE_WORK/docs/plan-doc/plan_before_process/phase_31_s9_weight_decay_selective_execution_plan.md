# Phase 31 S9 Weight-Decay Selective Execution Plan

## 1. Plan identity

```text
Plan ID: CW-PHASE-31-S9-SELECTIVE-EXECUTION-v1
Related issue: CW-PHASE-31-UPSTREAM-GATE-001
Phase detail: Phase_31_S9_Weight-decay_sweep.md
Status: HUMAN_APPROVED_FOR_IMPLEMENTATION
Scientific execution: CONDITIONAL_ON_CANONICAL_GATE
```

## 2. Objective

Implement Phase 31 as a clean terminal-owned controlled sweep extension while keeping `CourseWork.ipynb` presentation-only.

The implementation must recover state from validated files after restart, avoid rerunning the notebook and avoid invoking upstream materializers. Processing logs may be read for presentation continuity, but canonical Phase 30 artifacts remain authoritative for scientific execution.

## 3. Phase 31 contract

```text
Sweep ID: S9_WEIGHT_DECAY
Version: SWEEP_S9_WEIGHTDECAY-v1
Factor: AdamW weight_decay
WD0: 0.0
WD1: 0.0001
WD2: 0.001
Reference condition: WD1
Primary metric: Validation RMSE Wh
Tie rule: lower WD on exact RMSE tie
Test access: forbidden
```

WD1 may only reuse the exact S8 winner. WD0 and WD2 may only execute as fresh seed-42 runs after all Phase 30, environment, optimizer-group and Test-firewall gates pass.

## 4. Ownership

### 4.1. `src/course_work/experiments/phase_execution.py`

- Phase 31 dependency registry;
- exact WD condition registry;
- state and action resolution;
- expected-minus-verified calculation;
- execution readiness;
- complete block-reason propagation.

### 4.2. `src/course_work/sweeps/weight_decay.py`

- Phase 31 identity and frozen contract;
- canonical Phase 30 handoff validation;
- processing-log observation without treating it as canonical evidence;
- optimizer weight-decay condition mapping;
- WD1 reuse eligibility gate;
- Phase 31 condition preparation;
- Phase 31 terminal execution boundary;
- Test firewall declaration.

### 4.3. `src/course_work/reporting/phase_summary.py`

- Phase 31 derived log;
- compact HTML execution decision;
- prerequisite table;
- WD0, WD1 and WD2 evidence table;
- explicit block reason;
- aggregate-log inclusion.

### 4.4. Terminal scripts

- accept Phase 31 and `S9_WEIGHT_DECAY`;
- dispatch only missing authorized conditions;
- never materialize Phase 0 to Phase 30;
- never create header-only or fabricated result artifacts.

### 4.5. Notebook

Add a Phase 31 markdown section and one code cell:

```python
from course_work.reporting.phase_summary import render_phase_resume
render_phase_resume(31)
```

No CSV loading, training, plotting, branching or artifact writing may be added to the notebook.

## 5. Sequential implementation plan

### Step 1. Capture Phase 31 notebook baseline

Record current cell count, IDs, outputs and notebook checksum.

Verify notebook JSON and output hashes before edits.

### Step 2. Amend architecture scope

Extend current implementation scope through Phase 31 and register `sweeps/weight_decay.py` as the Phase 31 scientific owner.

Verify no ownership conflict with generic sweep results, execution or reporting modules.

### Step 3. Fix generic block-reason propagation

Expose exact prerequisite, signoff, artifact and environment reasons for all blocked selective phases.

Verify current Phase 30 HTML identifies the missing Phase 29 artifacts.

### Step 4. Register Phase 31

Add exact WD0, WD1 and WD2 values, Phase 30 prerequisites, S9 artifact paths and WD1 reference identity.

Verify unsupported values and unknown conditions fail.

### Step 5. Implement Phase 31 module

Create source-owned preflight and condition preparation APIs.

Verify:

- selected learning rate is loaded from S8 winner;
- no fallback learning rate exists;
- WD0 remains AdamW with zero weight decay;
- WD1 is reuse-only;
- WD2 is a fresh-run condition;
- Test remains forbidden;
- invalid Phase 30 blocks before model, loader or optimizer construction.

### Step 6. Extend terminal dispatch

Add Phase 31 to missing-only runner and background dry-run.

Verify Phase 31 dry-run resolves only Phase 31 command and actual execution gate blocks without creating a run.

### Step 7. Add JSON log and HTML

Run the read-only Phase 31 renderer to create:

```text
docs/save_log_in_processing/phase_31_s9_weight_decay_log.json
```

Verify static HTML has no widget MIME, JavaScript or external asset and includes exact block reasons.

### Step 8. Add notebook presentation

Append only the Phase 31 markdown and public reporting call before the aggregate-log cell.

Verify all previous cell sources and outputs remain unchanged and no Phase 31 processing logic enters the notebook.

### Step 9. Run tests

Add unit, contract and notebook-boundary coverage for Phase 31.

Verify compilation, focused tests, notebook validation, output preservation and dry-run behavior.

### Step 10. Scientific execution gate

If Phase 30 canonical evidence is valid, run only missing WD0 and WD2 conditions from terminal and reuse exact-match WD1.

If Phase 30 remains invalid, stop with Phase 31 `UPSTREAM_INVALID/BLOCK` and do not train.

## 6. Allowed file impact

```text
docs/RULE_BASE/architecture_rule.md
docs/plan-doc/analysis_error/phase_31_upstream_and_reason_visibility_issue.md
docs/plan-doc/plan_before_process/phase_31_s9_weight_decay_selective_execution_plan.md
docs/save_log_in_processing/phase_31_s9_weight_decay_log.json
notebook_course_work/CourseWork.ipynb
src/course_work/experiments/phase_execution.py
src/course_work/sweeps/weight_decay.py
src/course_work/sweeps/sweep_results.py
src/course_work/reporting/phase_summary.py
scripts/run_single_condition.py
scripts/run_all_pending.py
scripts/run_phase_background.py
scripts/run_phase_background/_notebook.py
tests/contracts/test_selective_execution_policy.py
tests/unit/test_phase_execution.py
tests/unit/test_selective_phase_reporting.py
tests/unit/test_weight_decay.py
tests/integration/test_notebook_boundary.py
```

## 7. Forbidden changes

- no raw-data changes;
- no Test access;
- no fabricated Phase 30 or Phase 31 result;
- no overwrite of signed invalid history;
- no hard-coded S8 learning-rate winner;
- no explicit L2 loss;
- no optimizer regrouping;
- no new bias or LayerNorm exclusion;
- no WD schedule;
- no WD1 retraining;
- no notebook training;
- no full notebook rerun;
- no global output clearing.

## 8. Current expected outcome

```text
Phase 31 implementation: available
Phase 31 processing log: generated
Phase 31 HTML: generated
Phase 31 scientific execution: blocked
Block state: UPSTREAM_INVALID
Canonical cause: Phase 30 winner and reference update missing because Phase 29 is incomplete
```

## 9. Approval record

```text
Human approved Phase 31 implementation: true
Human approved architecture extension: true
Human approved notebook presentation cell: true
Human requested terminal-only scientific execution: true
Scientific execution remains subject to canonical prerequisite gate: true
```
