# Phase 34 Current-State Rebaseline Blockers

> **Historical audit snapshot — superseded for current state.** Phase 34–41 đã
> được phục hồi/xác minh theo fail-closed legacy contracts và hiện đều là
> `VALID_REUSABLE` với action `RENDER_ONLY`. Giữ tài liệu này để truy vết tình
> trạng trước recovery; không dùng các kết luận `BLOCKED` bên dưới làm trạng
> thái hiện hành. Xem
> [FINAL_CONSISTENCY_AUDIT_20260913.md](../current_flow/FINAL_CONSISTENCY_AUDIT_20260913.md).

## Issue identity

```text
Issue ID: CW-PHASE-34-REBASELINE-001
Phase: 34
Phase name: S12 Head Sweep
Observed state: BLOCKED
Investigation scope: Phase 34 and its direct Phase 33 handoff
Scientific execution performed: false
Test access performed: false
```

## User objective

Rebaseline Phase 34 from the current repository state, preserve the S12 scientific contract, validate the implementation and stop before expensive H2 training.

## Expected behavior

Phase 34 may become ready for H2 only when all of the following are true:

```text
Phase 33 handoff is checksum-valid.
H4 exact reference evidence is complete and reusable.
The signed runtime environment is available.
H2 and H4 differ only in num_heads and derived head geometry.
Focused Phase 34 tests pass.
Audit-only and dry-run write no scientific artifact.
Test remains forbidden.
```

## Current repository evidence

The Phase 33 sweep handoff JSON files exist and match the checksums recorded in Phase 33:

```text
artifacts/sweeps/S11_d_model/s11_d_model_winner.json
sha256 0882ffd94a9867b11979559addc4ef20e4d485cc97eae724b88ea1c66608b780

artifacts/sweeps/S11_d_model/s11_reference_update.json
sha256 f8bd5da635a68529641ffde503047b277907358639b08c3156eff04a7f1e6dd4

artifacts/sweeps/S11_d_model/phase_33_signoff.json
sha256 f8caf290991f8a26fe1f899a0490a8727d2d8096172dc612b55571741fa7e76e
```

The Phase 33 handoff declares:

```text
status = PASS
approved_for_phase34 = true
winner_run_id = RUN_TR_S09_0016_AE0FB819
selected_d_model = 64
num_heads = 4
num_layers = 2
ffn_dim = 128
validation_rmse_wh = 58.08190056355405
test_status = FORBIDDEN
```

The following H4 files still exist and match registry checksums:

```text
artifacts/runs/RUN_TR_S09_0016_AE0FB819/config.json
artifacts/runs/RUN_TR_S09_0016_AE0FB819/status.json
artifacts/runs/RUN_TR_S09_0016_AE0FB819/metrics/best_validation_metrics.json
```

## Confirmed retention limitation: incomplete H4 canonical evidence

The experiment registry records H4 as `COMPLETED`, but the following declared or Phase-34-required files are absent:

```text
artifacts/runs/RUN_TR_S09_0016_AE0FB819/checkpoints/best_checkpoint.pt
artifacts/runs/RUN_TR_S09_0016_AE0FB819/training_history.csv
artifacts/runs/RUN_TR_S09_0016_AE0FB819/training.log
artifacts/runs/RUN_TR_S09_0016_AE0FB819/predictions/best_validation_predictions.csv
```

Expected checksums from the registry and preservation baseline:

```text
best_checkpoint.pt
3ccf735488336340275a4dccf040fcd17b98a4fa4746d3e665cf14f97428d75d

training_history.csv
635e00dfcd032c3664c5e031b13f38d94caa3f73c4c6bf142eb0d6883d9eb43b

training.log
f709354699e8fb103b226fbae64ac2d2611fd80854ae1342246e86c94415d5d7

best_validation_predictions.csv
9bb212edb469f8fac02cf9179a401fec04d27487131a1e17a5e14179b94a5a4e
```

Repository recovery audit found:

```text
No duplicate best_checkpoint.pt exists under COURSE_WORK.
No duplicate training_history.csv exists under COURSE_WORK.
The missing files are ignored by .gitignore.
The missing files were never tracked by Git.
No Git LFS object is registered for the H4 run.
```

The original files are confirmed unrecoverable. Human approved a corrective scientific protocol that accepts H4 only as:

```text
HISTORICAL_REFERENCE_WITH_INCOMPLETE_ARTIFACT_RETENTION
```

H4 must not be retrained, reconstructed from metrics or replaced with a stub. The retained evidence must be cross-validated, the missing paths must remain explicitly missing and Phase 34 must finish as `PASS_WITH_WARNING` if H2 later completes and every remaining contract passes.

## Confirmed blocker 2: signed execution environment unavailable

The signed environment requires:

```text
Python version: 3.10.11
Python executable: /Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10
Project root: /Users/vientu/Deep Learning/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK
Selected device: mps
```

Current repository runtime provides:

```text
.venv Python version: 3.11.14
.venv package course-work: not installed
signed Python 3.10 executable: missing
```

The required import currently fails before preflight:

```text
ModuleNotFoundError: No module named 'course_work'
```

Installing the package into Python 3.11 would conflict with the declared `requires-python = ">=3.10,<3.11"` contract and would not repair the signed environment identity.

## Confirmed stale or missing Phase 34 records

The following tracked Phase 34 files are deleted from the current working tree:

```text
docs/plan-doc/plan_before_process/phase_34_s12_head_sweep_selective_terminal_execution_plan.md
docs/save_log_in_processing/phase_34_implementation_preservation_baseline.json
docs/save_log_in_processing/phase_34_regression_preservation_baseline.json
docs/save_log_in_processing/phase_34_s12_head_log.json
```

Their committed versions describe the former state and are not authoritative for the current working tree. The former processing log reported `CONDITION_INCOMPLETE`, H2 missing, H4 reused and `SCIENTIFIC_EXECUTION_NOT_AUTHORIZED`, but the current H4 evidence loss makes that log stale.

No canonical Phase 34 artifact root currently exists:

```text
artifacts/sweeps/S12_heads/
```

No H2 scientific run is present.

## Phase 34 implementation observations

Phase-34-owned implementation is present in:

```text
src/course_work/sweeps/heads.py
src/course_work/experiments/phase_execution.py
src/course_work/experiments/sweep_recovery.py
src/course_work/sweeps/sweep_results.py
src/course_work/reporting/phase_summary.py
scripts/run_single_condition.py
scripts/run_all_pending.py
```

The requested `src/course_work/recovery/` package does not exist. This is not itself a missing dependency because the approved architecture assigns recovery ownership to `src/course_work/experiments/sweep_recovery.py`.

Current code correctly declares H2 as `TRAIN_NEW`, H4 as `REUSE_REFERENCE`, selected `d_model` dynamically from Phase 33, exact-tie preference H2 and Test forbidden. `prepare_phase_34_condition` rejects an invalid Phase 33 handoff before training.

The approved corrective implementation now covers the previously identified Phase 34 gaps:

```text
The Phase 34 finalizer emits several compact audit tables whose columns do not yet cover the full Phase-detail schemas.
The generated Phase 34 sign-off does not yet include every minimum field listed by the Phase detail.
The current focused finalizer test checks required file presence but not every required schema field.
The H2 runner now rejects every config delta except `model.num_heads: 4 -> 2` before loader, registry, model or optimizer construction.

The H4 resolver now accepts only `HISTORICAL_REFERENCE_WITH_INCOMPLETE_ARTIFACT_RETENTION`, emits `PASS_WITH_WARNING`, records all four missing paths and keeps any mismatch in retained evidence as a hard block.

Phase 34 finalization now requires a valid retained-evidence handoff, emits `PASS_WITH_WARNING` and propagates the retention warning and missing-artifact metadata into the Phase 35 reference update.

Focused runtime tests and the official Phase 34 audit-only entrypoint remain blocked because the signed Python 3.10.11 interpreter is unavailable. Static compilation and the read-only seven-source evidence audit pass.
Checkpoint metadata contains the model config but does not explicitly record derived head_dim.
```

These are implementation-verification items, not permission to fabricate missing H4 evidence.

## Notebook evidence

The current notebook contains the Phase 34 presentation-only cells:

```text
phase-34-heading
phase-34-resume
```

The code cell only calls:

```python
from course_work.reporting.phase_summary import render_phase_resume
render_phase_resume(34)
```

Both cells currently have no stored outputs. The notebook is already dirty and must not be modified during the blocked rebaseline stage.

## Root cause

Phase 34 is blocked by lost, Git-ignored H4 run evidence and an unavailable signed Python environment. The old Phase 34 processing state assumed those dependencies were present. The repository can still identify the expected files and checksums, but it cannot reconstruct their scientific contents.

## Upstream impact

Phase 33 JSON selection evidence remains readable, but its selected H4 run is not fully reproducible or checkpoint-verifiable from the current repository state.

## Downstream impact

```text
H2 training authorization must remain false.
Phase 34 winner selection is forbidden.
Phase 34 sign-off is forbidden.
Phase 35 handoff is forbidden.
Notebook Phase 34 can only display a blocked state after the runtime environment is repaired.
```

## Risk level

```text
Scientific integrity risk: critical
Artifact integrity risk: critical
Source-code regression risk: medium
Notebook risk: low when left untouched
```

## Files that must not change

```text
Phase 33 winner, reference and sign-off JSON
RUN_TR_S09_0016_AE0FB819 existing config, status and metrics
Experiment registry historical records
Any Test artifact or Test-access state
CourseWork.ipynb outside an approved Phase 34 presentation change
Any Phase outside the direct Phase 33 to Phase 34 dependency
```

## Approved correction direction

Implement the Human-approved historical-reference validator from the retained Phase 33 sweep artifacts, Experiment Registry, H4 config, status and Validation metrics. Preserve the four missing paths as explicit missing-retention warnings. Do not retrain H4. Propagate the warnings into the Phase 34 winner, sign-off, summary, report, processing log and Phase 35 handoff. Separately repair the runtime only through the official environment recovery mechanism.

## Current conclusion

```text
PHASE_34_TRUE_STATE = CORRECTIVE_PROTOCOL_APPROVED
H2 = NOT_STARTED_AND_NOT_AUTHORIZED
H4 = HISTORICAL_REFERENCE_WITH_INCOMPLETE_ARTIFACT_RETENTION
PREFLIGHT = BLOCKED
TEST = FORBIDDEN_AND_NOT_ACCESSED
```

## Human protocol approval

```text
Approval date: 2026-08-24
Original H4 artifact restoration: TERMINATED_AS_UNRECOVERABLE
H4 retraining: FORBIDDEN
Historical reference mode: APPROVED
Required Phase 34 final status: PASS_WITH_WARNING
Warning propagation to Phase 35: REQUIRED
H2 execution by agent: FORBIDDEN
```
