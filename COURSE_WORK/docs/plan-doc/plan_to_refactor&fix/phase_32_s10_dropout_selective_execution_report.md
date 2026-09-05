# Phase 32 S10 Dropout Selective Execution Report

## 1. Scope

This report records the implementation and verification of Phase 32 under the approved selective terminal execution plan.

The implementation preserves the project flow:

- scientific execution belongs to Python modules and terminal entry points;
- canonical artifacts remain authoritative;
- processing logs remain derived presentation records;
- the notebook only calls a public reporting function;
- Test access remains forbidden;
- no Phase 31 or Phase 32 scientific result is fabricated.

## 2. Implemented architecture

Phase 32 scientific ownership is located at:

```text
src/course_work/sweeps/dropout.py
```

Selective inspection, execution routing and background dispatch are located at:

```text
src/course_work/experiments/phase_execution.py
scripts/run_single_condition.py
scripts/run_all_pending.py
scripts/run_phase_background.py
scripts/run_phase_background/_notebook.py
```

Result conversion and canonical finalization are located at:

```text
scripts/sweep_results_to_csv.py
src/course_work/sweeps/sweep_results.py
```

Derived reporting is located at:

```text
src/course_work/reporting/phase_summary.py
docs/save_log_in_processing/phase_32_s10_dropout_log.json
```

Notebook presentation is located at:

```text
notebook_course_work/CourseWork.ipynb
```

## 3. Registered S10 contract

```text
DR01 = 0.1
DR02 = 0.2
DR03 = 0.3
Reference condition = DR01
Dropout scope = all Transformer encoder dropout sites
Optimizer = AdamW
Loss = MSE
Maximum epochs = 50
Early-stopping patience = 10
Gradient clipping = 1.0
Seed = 42
Primary selection metric = Validation RMSE Wh
Exact tie rule = lower dropout
MC Dropout = disabled
Test access = forbidden
```

The selected weight decay and learning rate are not hard-coded. They must be loaded from a valid and internally consistent Phase 31 winner and reference handoff.

## 4. Sequential implementation result

### Step 1. Notebook preservation baseline

Status: PASS

The baseline records 96 existing cells. After Phase 32 integration, the notebook contains 98 cells.

Verification result:

```text
New cells = phase-32-heading, phase-32-resume
Changed existing cell IDs = none
Changed existing cell sources = none
Changed existing cell outputs = none
Changed existing execution counts = none
Phase 32 position = after Phase 31 and before aggregate logs
```

### Step 2. Architecture authorization

Status: PASS

The architecture scope now includes Phase 32. The Phase 32 owner, artifact directory, processing log, selective terminal route and notebook boundary are registered.

### Step 3. Dropout scope and runtime semantics

Status: PASS

The Phase 32 owner validates four controlled dropout sites per encoder layer:

```text
attention-weight dropout
attention-residual dropout
FFN hidden dropout
FFN residual dropout
```

The current two-layer Transformer therefore has eight controlled sites. The audit verifies a common global probability, stable topology fingerprint, stochastic train behavior, deterministic eval behavior and no MC Dropout.

### Step 4. Phase 31 handoff gate

Status: PASS

The gate validates:

- Phase 31 sign-off status and explicit Phase 32 approval;
- Test-firewall state;
- winner and reference run identity;
- selected weight decay consistency;
- frozen feature, target-scaling, lookback, pooling, activation, batch and population fields;
- exact winner run configuration and fingerprint;
- D64, H4, N2 and FFN128 capacity;
- AdamW, MSE, epoch budget, patience, clipping, scheduler and seed;
- dropout 0.1 and the eight-site scope;
- unique canonical experiment-registry evidence before DR01 reuse.

The Phase 31 processing log is recorded as non-authoritative observational input only.

### Step 5. Terminal execution boundary

Status: PASS

Phase 32 can be selected independently:

```text
python scripts/run_phase_background.py sweep --phase-id 32 --dry-run
```

The dry-run resolves only:

```text
python scripts/run_all_pending.py --phase-id 32 --dry-run
```

DR01 reuses the exact verified Phase 31 winner without retraining. DR02 and DR03 load the exact Phase 31 winner configuration and change only `model.dropout` after the execution gate passes.

### Step 6. Canonical finalization contract

Status: PASS

Canonical finalization requires all three conditions to have complete registry evidence and full Validation MAE, RMSE and R2. Winner selection uses full-precision Validation RMSE Wh and lower dropout only for an exact tie.

The finalizer generates the canonical metrics table, manifest, winner, Phase 33 reference update and signed output checksums only after condition completeness is verified.

### Step 7. Derived JSON and HTML presentation

Status: PASS

The Phase 32 log contains:

- execution decision;
- prerequisite validation;
- frozen S10 contract;
- DR01, DR02 and DR03 evidence;
- explicit block reasons;
- canonical source checksums when source files exist.

The HTML is static and contains no JavaScript, external resource or Jupyter widget payload.

### Step 8. Notebook integration

Status: PASS

The notebook code cell contains only:

```python
from course_work.reporting.phase_summary import render_phase_resume
render_phase_resume(32)
```

No Phase 32 processing, training, selection or JSON parsing logic was added to the notebook.

## 5. Plan adjustments made during implementation

### Adjustment 1. Explicit dropout topology audit

The implementation records each controlled encoder dropout site and a stable scope fingerprint instead of validating only the scalar configuration field.

### Adjustment 2. Shared finalizer extension through Phase 32

The shared canonical finalizer now carries learning rate, weight decay, dropout and capacity identity through Phase 31 and Phase 32 handoffs. This allows Phase 32 to validate the exact Phase 31 configuration without hard-coded winner values.

### Adjustment 3. Canonical Phase 32 terminal paths

Terminal result conversion and summary persistence now use:

```text
artifacts/sweeps/S10_dropout/live_sweep_results.jsonl
artifacts/sweeps/S10_dropout/s10_dropout_metrics.csv
```

### Adjustment 4. Independent Phase 32 execution

Dependency mode does not silently execute Phase 31 from the Phase 32 command. Phase 32 audits its immediate canonical handoff and blocks when that handoff is incomplete.

## 6. Current scientific execution state

Status: BLOCKED

Phase 32 training was not started because the current workspace does not provide a canonical Phase 31 handoff.

Current critical evidence gaps:

```text
artifacts/sweeps/S9_weight_decay/phase_31_signoff.json = missing
artifacts/sweeps/S9_weight_decay/s9_weight_decay_winner.json = missing
artifacts/sweeps/S9_weight_decay/s9_reference_update.json = missing
```

The terminal execution gate therefore returns:

```text
State = UPSTREAM_INVALID
Effective action = BLOCK
Missing conditions = DR01, DR02, DR03
Training runs created = 0
Scientific Phase 32 artifact root created = false
Experiment registry changed = false
Phase 31 processing log changed = false
Test accessed = false
```

## 7. Verification evidence

Focused Phase 32 owner verification:

```text
Module compile = PASS
Dropout sites = 8
Dropout scope = PASS
Train/eval semantics = PASS
```

Focused Phase 32 and selective execution tests:

```text
32 passed
```

Finalization tests:

```text
4 passed
```

Reporting, notebook contract and background runner tests:

```text
17 passed
```

Integrated Phase 32 regression set:

```text
51 passed
```

Full unit and contract suite:

```text
196 passed
9 failed
6 errors
4 subtests passed
```

The full-suite failures are outside the Phase 32 change set and occur in existing upstream environment, EDA, Phase 17, registry, persistence and chronological-split layers. They include GPU enforcement in the current process and invalid existing registry or artifact evidence. They were not bypassed or modified during Phase 32 implementation.

Notebook preservation verification:

```text
JSON structure = valid
Cell count = 98
New cells = 2
Changed existing cell IDs = 0
Changed existing sources = 0
Changed existing outputs = 0
Changed existing execution counts = 0
Widget MIME outputs introduced = 0
Embedded script elements introduced = 0
```

## 8. Completion state

```text
Phase 32 code architecture = complete
Selective terminal routing = complete
Read-only gate = complete
Dropout scope verification = complete
Canonical finalizer route = complete
Derived JSON log = complete
Static HTML visualization = complete
Notebook presentation = complete
Scientific S10 sweep = blocked by missing Phase 31 canonical handoff
Phase 33 scientific execution = remains locked
```
