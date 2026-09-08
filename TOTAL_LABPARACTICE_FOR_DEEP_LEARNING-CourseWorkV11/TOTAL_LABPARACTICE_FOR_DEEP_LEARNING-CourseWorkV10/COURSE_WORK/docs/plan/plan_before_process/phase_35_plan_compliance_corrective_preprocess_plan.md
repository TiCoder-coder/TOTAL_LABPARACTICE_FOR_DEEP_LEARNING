# Phase 35 — Plan Compliance Corrective Pre-Process Plan

## Plan identity

- Plan ID: `CW-PHASE35-COMPLIANCE-FIX-001`
- Phase ID: 35
- Phase name: S13 Layer Sweep
- Issue reference: `CW-PHASE35-COMPLIANCE-001`
- Execution state: `COMPLETED`
- Scientific execution: `FORBIDDEN`
- Test access: `FORBIDDEN`
- Phase 36 execution: `FORBIDDEN`

## Objective

Bring the finalized Phase 35 evidence package into full contract compliance with `Phase_35_S13_Layer_sweep.md` without retraining, changing scientific results, accessing Test, fabricating N2 evidence, or executing Phase 36.

## Frozen scientific result

- N1: `RUN_TR_S13_0021_9CA63891`
- N1 Validation RMSE: `59.78262924121282` Wh
- N2: `RUN_TR_S09_0016_AE0FB819`
- N2 Validation RMSE: `58.08190056355405` Wh
- Winner: `N2`
- Selected `num_layers`: `2`
- Required final status: `PASS_WITH_WARNING`
- Inherited warning: `H4_SOURCE_ARTIFACT_RETENTION_INCOMPLETE`

## Upstream dependencies

- Phase 34 winner, reference update, and sign-off
- N1 completed registry record and retained run artifacts
- N2 historical registry/config/status/metrics evidence
- Phase 9 validated scalers
- Phase 10 WINDOWPOP-v1 and window indices
- Phase 11 DataLoader contract
- Phase 12 METRICS-v1 contract
- Phase 13 Experiment Registry
- Transformer-v1 and Training Engine contracts

## Downstream dependencies

- Phase 36 consumes `s13_layer_winner.json` and `s13_reference_update.json`
- Phase 35 notebook presentation consumes canonical Phase 35 artifacts through `reporting/phase_summary.py`
- Selective execution consumes Phase 35 manifest, sign-off, result, winner, and reference artifacts

## Related files read before this plan

- `working_rule.md`
- `docs/RULE_BASE/architecture_rule.md`
- `docs/RULE_BASE/rule_code.md`
- `docs/plan-doc/plan_detail_for_each_phase/Phase_35_S13_Layer_sweep.md`
- `docs/plan-doc/plan_before_process/phase_35_s13_layer_sweep_preparation_plan.md`
- `docs/plan-doc/analysis_error/phase_35_plan_compliance_gap_analysis.md`
- `docs/plan-doc/plan_to_refactor&fix/phase_35_plan_compliance_corrective_plan.md`
- `src/course_work/sweeps/layers.py`
- Phase-35-owned sections of `src/course_work/sweeps/sweep_results.py`
- Phase-35-owned sections of `src/course_work/experiments/phase_execution.py`
- Phase-35-owned sections of `src/course_work/reporting/phase_summary.py`
- Validation/loading paths in `src/course_work/training/engine.py`, `src/course_work/data/datasets.py`, and `src/course_work/data/scaling.py`
- Phase-35-owned paths in `scripts/run_single_condition.py` and `scripts/run_all_pending.py`
- `tests/unit/test_layers.py`
- Phase-35-owned sections of `tests/unit/test_sweep_finalization.py`, `tests/unit/test_phase_execution.py`, and `tests/unit/test_selective_phase_reporting.py`
- Phase 35 notebook cells in `notebook_course_work/CourseWork.ipynb`
- Current Phase 35 canonical artifacts and N1/N2 run evidence

## Files planned for modification

- `src/course_work/sweeps/layers.py`
- `src/course_work/sweeps/sweep_results.py`
- `tests/unit/test_layers.py`
- `tests/unit/test_sweep_finalization.py`
- `tests/contracts/test_phase_35_compliance.py` if a dedicated contract test is required
- `src/course_work/reporting/phase_summary.py` only if corrected canonical fields require a presentation adapter update
- `notebook_course_work/CourseWork.ipynb` only if the Phase 35 presentation cell/output must be refreshed
- `docs/save_log_in_processing/phase_35_s13_layer_log.json`
- Canonical Phase 35 artifacts under `artifacts/sweeps/S13_layers/`

## Files and state forbidden from modification

- `artifacts/runs/RUN_TR_S13_0021_9CA63891/checkpoints/*.pt`
- All N1 model weights and scientific metrics
- All N2 historical run files and missing-artifact declarations
- Phase 34 winner, reference, sign-off, results, and checksums
- Experiment Registry scientific result/config records
- Phase 9–13 scientific contracts
- Training Engine scientific behavior
- Raw/interim data
- Any Test target, Test metric, Test prediction, or Test iterator
- Phase 36 source, artifacts, registry entries, or execution state

## Architecture constraints

- Reusable processing remains under `src/course_work/`.
- Phase 35 geometry/preflight logic remains in `sweeps/layers.py`.
- Canonical Phase 35 output construction remains in `sweeps/sweep_results.py`.
- Notebook remains presentation/orchestration only.
- Machine-readable processing log remains JSON under `docs/save_log_in_processing/`.
- Raw terminal logs remain `.log` under `artifacts/sweeps/logs/`.
- No source comments, decorative symbols, new dependency, absolute path, or notebook processing logic may be introduced.
- Artifact writes must be atomic, checksum-verified, and idempotent.

## Phase requirement mapping

| Phase Detail requirement | Implementation owner | Verification evidence |
|---|---|---|
| Sections 99–103 N1 BEST and provenance | `sweeps/layers.py`, `sweeps/sweep_results.py` | strict-load full ordered Validation result, provenance fields, focused tests |
| O35.1–O35.4 manifest/contract/preflight/run matrix | `sweeps/sweep_results.py` | exact field/schema tests |
| O35.5–O35.13 geometry/parameter/optimizer audits | `sweeps/layers.py`, `sweeps/sweep_results.py` | per-layer/per-parameter CSV schema tests and runtime assertions |
| O35.14 config delta | `sweeps/sweep_results.py` | exact only-`model.num_layers` comparison |
| O35.15 Section 164 unit-test evidence | `tests/unit/test_layers.py`, finalizer artifact builder | every prescribed case represented and passing |
| O35.16 common data | `sweeps/sweep_results.py` | Train/Validation population and frozen-field rows; unavailable N2 ordering remains explicit |
| O35.17–O35.24 training/init/order/attention/budget/provenance | `sweeps/layers.py`, `sweeps/sweep_results.py` | exact CSV schema and evidence-status tests |
| O35.25 N2 reuse | existing Phase 34 historical-reference gate | no retraining and warning propagation tests |
| O35.26 verified N1 | read-only BEST verifier | recomputed METRICS-v1 equality within contract tolerance |
| O35.27–O35.38 metrics through Phase36 reference | `sweeps/sweep_results.py` | exact schemas, winner frozen, policy fields |
| O35.39 figures | existing artifact-driven renderer | required PNG existence/checksum tests |
| O35.40–O35.45 tests/log/summary/report/README/sign-off | `sweeps/sweep_results.py` | detailed content/schema and sign-off validation |
| Section 218 minimum sign-off fields | `sweeps/sweep_results.py` | required-key contract test |
| Section 185/186 Phase36 handoff | `sweeps/sweep_results.py` | `F64=TRAIN_NEW`, `F128=REUSE_REFERENCE`, `F256=TRAIN_NEW` assertions |
| Test firewall | all owners | no Test iterator/metric/prediction evidence and focused firewall tests |

## Current project state

- Phase 34: `PASS_WITH_WARNING`, approved for Phase 35.
- N1 and N2 scientific evidence: finalized and frozen.
- Phase 35 result: scientifically consistent but contract-incomplete.
- Existing canonical output checksums: internally consistent with the abbreviated outputs.
- Phase 36: must remain blocked until corrective compliance passes.

## Preconditions before implementation

1. Human explicitly approves this exact pre-process plan.
2. Exact interpreter remains `/Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10`.
3. Phase 34 handoff remains checksum-valid.
4. N1 checkpoint and Validation-only artifacts remain checksum-valid.
5. N2 historical evidence remains in the approved incomplete-retention mode.
6. No other active execution task modifies Phase 35-owned files.

## Known risks

- Full Validation recomputation can expose deterministic floating-point differences; tolerances must come from METRICS-v1 or established verification policy and must not be weakened merely to pass.
- Calling existing materializers from the verifier may write upstream artifacts; the implementation must use validated read paths without mutating upstream scientific artifacts.
- Replacing signed Phase 35 outputs changes their checksums; replacement must use the canonical preserved-replacement path and regenerate the Phase 35 sign-off atomically.
- Shared finalizer changes can regress Phase 23–34; Phase 35-specific behavior must be isolated where possible.
- N2 history, checkpoint, predictions, and sample-order evidence cannot be reconstructed or fabricated.
- Notebook refresh must not introduce processing or hidden state.

## Sequential implementation plan

### Step 1 — Baseline and immutable evidence guard

Capture hashes for frozen N1/N2 scientific inputs, Phase 34 handoff, Registry, notebook, and current Phase 35 outputs. Verify Test remains locked.

Stop if any frozen scientific input differs from the approved baseline.

### Step 2 — Expand Phase 35 runtime audit model

Extend `sweeps/layers.py` with deterministic in-memory evidence for exact layer definitions, stack geometry, architecture roles, state-dict delta, shared schemas, parameter counts, layer independence, optimizer coverage, initialization policy, attention API, PE policy, and prohibited-feature assertions.

Run `tests/unit/test_layers.py` immediately after this logical change.

### Step 3 — Add read-only N1 BEST verifier

Build a fresh frozen N1 model, strict-load the retained BEST checkpoint, use `eval()`, iterate only the complete ordered Validation population, inverse-transform YS1 using the validated Train-only target scaler, compute METRICS-v1, and compare RMSE/MAE/R2/sample count/population fingerprint with stored N1 BEST evidence.

Do not call `Trainer.train()`, optimizer steps, Test loaders, or Test evaluation APIs.

Run focused verifier tests immediately.

### Step 4 — Generate exact O35 schemas

Replace abbreviated Phase 35 payload construction with exact field sets from Sections 150–185. Preserve explicit unavailable values/statuses for historical N2 evidence. Do not substitute summary booleans where per-layer or per-parameter rows are required.

Run Phase 35 finalizer fixture tests immediately.

### Step 5 — Complete version, provenance, sign-off, and handoff

Emit `SWEEP_S13_LAYERS-v1`, complete checkpoint/run provenance, exact Section 218 sign-off fields, exact Section 185 reference fields, and explicit Phase 36 condition policy. Preserve winner, metrics, inherited warning, and Test status.

Run sign-off, checksum, version, and handoff contract tests immediately.

### Step 6 — Regenerate canonical Phase 35 outputs

Use only the canonical finalization replacement path. Archive/preserve stale Phase 35 outputs according to existing artifact policy, write corrected outputs atomically, recompute output checksums, and validate the new sign-off.

Do not mutate run artifacts, Registry scientific records, Phase 34, or Phase 36.

### Step 7 — Refresh derived presentation only if required

Refresh `phase_35_s13_layer_log.json` from corrected canonical evidence. Update notebook presentation only if its current public API call cannot display the corrected evidence. No processing logic may enter the notebook.

### Step 8 — Validation and compliance audit

Run focused tests, audit-only, dry-run, finalization verification twice for idempotency, Test-firewall verification, artifact-schema validation, warning-propagation validation, notebook-boundary tests, and a fresh requirement-to-evidence compliance audit.

Stop before Phase 36 regardless of the result.

## Focused validation commands

Use exactly:

```bash
cd COURSE_WORK
MPLCONFIGDIR=/private/tmp/course-work-phase35-corrective \
PYTHONPATH=src \
/Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10 \
  -m pytest \
  tests/unit/test_layers.py \
  tests/unit/test_sweep_finalization.py \
  tests/unit/test_phase_execution.py \
  tests/unit/test_selective_phase_reporting.py \
  tests/integration/test_notebook_boundary.py -q
```

```bash
cd COURSE_WORK
PYTHONPATH=src /Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10 \
  scripts/run_all_pending.py --phase-id 35 --audit-only
```

```bash
cd COURSE_WORK
PYTHONPATH=src /Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10 \
  scripts/run_all_pending.py --phase-id 35 --dry-run
```

Finalization and idempotency commands will use the existing canonical finalizer API with the preserved-replacement flag only after the focused tests pass.

## Regression strategy

- Verify Phase 34 input hashes and handoff status before and after correction.
- Verify N1/N2 run artifact hashes and Registry scientific result records remain unchanged.
- Verify generic sweep sign-off validation still passes for the current Phase 34 handoff.
- Verify Phase 35 selective state becomes `VALID_REUSABLE` after derived log refresh.
- Verify Phase 36 is not executed and only receives corrected metadata.
- Verify no Test-target access, Test metric, Test prediction, or Test iterator evidence appears.
- Verify notebook remains presentation/orchestration only.

## Expected outputs

- Exact read-only N1 BEST verification evidence
- Complete O35.1–O35.45 mandatory artifacts
- Optional omissions documented exactly where Phase Detail permits them
- `SWEEP_S13_LAYERS-v1` identity
- Complete Section 218 sign-off
- Complete Section 185 reference update
- Explicit Phase 36 F64/F128/F256 policy
- Preserved `PASS_WITH_WARNING` inherited-warning semantics
- Passing focused tests and idempotent finalization validation

## Acceptance criteria

- Recomputed N1 Validation metrics match canonical stored metrics under the approved METRICS-v1 verification tolerance.
- Strict checkpoint load succeeds with frozen N1 config and required provenance.
- Every mandatory O35 artifact exists and satisfies its exact schema.
- All Section 164 unit-test cases are represented and pass.
- N1/N2 depth-only scientific contract remains unchanged.
- Parameter counts remain 35713 and 69185; delta remains 33472 and equals N2-only second-layer numel.
- N2 layers remain independent and optimizer coverage is complete without duplicates.
- Sign-off and reference update contain every required field.
- Warning propagation remains complete.
- Test remains forbidden and untouched.
- Phase 36 code/training is not executed.
- Fresh compliance audit reports `COMPLIANT_WITH_DOCUMENTED_WARNING` or stronger without changing the frozen winner.

## Rollback and stop conditions

Stop without broadening scope if:

- strict-load or full Validation recomputation does not match stored N1 evidence;
- a frozen scientific artifact checksum changes unexpectedly;
- Phase 34 handoff becomes invalid;
- exact O35 schema requires a change outside Phase-35-owned modules;
- a focused test exposes an unplanned shared-finalizer regression;
- Test access would be required;
- N2 missing historical evidence would need fabrication;
- Phase 36 execution would be triggered;
- any new dependency or protocol change is required.

No destructive repository-wide command is permitted. Any canonical replacement must remain limited to Phase 35 outputs and use the existing preserved-replacement mechanism.

## User approval gate

`USER_APPROVED_PREPROCESS_PLAN=true`

Human approval was granted for `CW-PHASE35-COMPLIANCE-FIX-001` before implementation.

## Execution result

- Corrective implementation status: `PASS_WITH_WARNING`
- N1 BEST strict-load verification: `PASS`
- Recomputed N1 Validation RMSE: `59.78262891370939`
- Stored N1 Validation RMSE: `59.78262924121282`
- Device-tolerance comparison: `PASS` at `1e-6`
- Required Phase 35 outputs: `COMPLETE`
- Sweep version: `SWEEP_S13_LAYERS-v1`
- Section 218 sign-off schema: `PASS`
- Phase 36 policy: `F64=TRAIN_NEW`, `F128=REUSE_REFERENCE`, `F256=TRAIN_NEW`
- Focused tests: `72 passed`
- Audit-only: `PASS`, state `VALID_REUSABLE`
- Dry-run: `PASS`, no scientific artifacts written
- Idempotent finalization: `PASS`
- Test access: `FORBIDDEN`
- Training executed: `NO`
- Phase 36 executed: `NO`
