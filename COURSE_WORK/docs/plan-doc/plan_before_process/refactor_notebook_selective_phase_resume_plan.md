# Refactor Notebook Selective Phase Resume Plan

## 1. Plan identity

- Plan ID: `CW-NOTEBOOK-SELECTIVE-RESUME-PLAN-001`
- Related issue: `CW-NOTEBOOK-SELECTIVE-RESUME-001`
- Scope: persistent notebook presentation and selective execution for late phases, with Phase 30 as the first controlled target
- Status: awaiting user approval

## 2. Objective

Refactor the workflow so that a selected phase can be inspected and resumed after a kernel restart without executing the preceding notebook cells. The workflow must reuse verified artifacts, reconstruct missing derived records, execute only genuinely missing conditions, and block when a prerequisite or environment is invalid.

Phase 30 is the first acceptance case. Its notebook cell must be independently runnable and must not trigger Phase 0 to Phase 29.

## 3. Non-goals

- Do not rerun the entire notebook.
- Do not clear or recreate all notebook outputs.
- Do not retrain conditions that have valid canonical evidence.
- Do not repair Phase 29 or run Phase 30 training under this planning approval alone.
- Do not change learning-rate values, selection metrics, phase order or Test-lock policy.
- Do not accept synthetic, bypassed or unverifiable results.

## 4. Mandatory architecture amendment

Before source code changes, update `docs/RULE_BASE/architecture_rule.md` for explicit user approval of:

- Phase 15 to Phase 30 module ownership;
- the selective phase execution service;
- canonical artifact versus derived log authority;
- condition-level resume behavior;
- immutable signed artifact revisions;
- notebook presentation allowlists;
- the rule that notebook cells call public APIs and contain no processing logic.

Implementation must stop if this amendment is not approved.

## 5. Proposed ownership

### 5.1 Selective execution service

Create `src/course_work/experiments/phase_execution.py` as the single owner of:

- phase dependency registry;
- expected condition registry;
- canonical evidence inspection;
- state classification;
- action resolution;
- missing-only dispatch;
- presentation-log regeneration;
- notebook-ready result construction.

This module must inspect prerequisite evidence without calling prerequisite materializers.

### 5.2 Existing module responsibilities

- `src/course_work/sweeps/sweep_results.py`: enforce artifact and checksum validation before reusing a signoff.
- `src/course_work/reporting/phase_summary.py`: render only approved, compact, persistent phase information.
- background and condition scripts: expose selective phase and condition execution without hidden upstream materialization.
- notebook: call one public function for Phase 30 and display its returned persistent output.

## 6. State model

Every inspected phase must resolve to exactly one state:

- `VALID_REUSABLE`: all required canonical artifacts, checksums, conditions and lineage are valid;
- `LOG_MISSING`: canonical evidence is valid but its derived processing log is absent;
- `LOG_STALE`: canonical evidence is valid but the derived log no longer matches it;
- `DERIVED_ARTIFACT_MISSING`: a reproducible derivative is absent while canonical run evidence is valid;
- `CONDITION_INCOMPLETE`: one or more expected conditions lack valid run evidence;
- `SIGNOFF_INVALID`: the signoff conflicts with files, checksums, configuration or lineage;
- `UPSTREAM_INVALID`: a required prior phase is incomplete or invalid;
- `ENVIRONMENT_INVALID`: execution cannot satisfy the signed environment contract;
- `RUNNING`: required work is already owned by an active process;
- `FAILED`: a prior attempted condition failed and is eligible for controlled retry.

## 7. Action model

The state resolver must choose the smallest safe action:

| State | Action |
|---|---|
| `VALID_REUSABLE` | `RENDER_ONLY` |
| `LOG_MISSING` | `REBUILD_LOG_ONLY` |
| `LOG_STALE` | `REBUILD_LOG_ONLY` |
| `DERIVED_ARTIFACT_MISSING` | `REBUILD_DERIVED_ONLY` |
| `CONDITION_INCOMPLETE` | `EXECUTE_MISSING_ONLY` |
| `RUNNING` | `WAIT_FOR_RUNNING_PROCESS` |
| `SIGNOFF_INVALID` | `BLOCK` or create an approved new artifact revision |
| `UPSTREAM_INVALID` | `BLOCK` |
| `ENVIRONMENT_INVALID` | `BLOCK` |

No action may invoke Phase 0 to Phase 29 merely because Phase 30 was selected.

## 8. Evidence trust hierarchy

Validation must follow this order:

1. approved phase detail and architecture rule;
2. approved configuration;
3. experiment registry run configuration and status;
4. run artifacts and checksums;
5. phase manifest and signoff;
6. processing log and its source checksums;
7. stored notebook output.

Lower levels cannot override contradictions at higher levels.

## 9. Phase 30 completeness contract

Phase 30 is complete only when all of the following are true:

- Phase 29 has a valid winner and reference-update artifact;
- expected conditions are exactly LR1 `0.0001`, LR2 `0.0003` and LR3 `0.001`;
- LR2 is reused from the approved Phase 29 reference without retraining;
- LR1 and LR3 each have valid, attributable run evidence;
- every run's canonical configuration matches its assigned learning rate;
- `results.csv`, winner, reference-update, manifest and signoff all exist and agree;
- all declared checksums match;
- winner selection uses Validation RMSE in Wh;
- Test remains locked;
- the processing log identifies the active valid artifact revision.

The current repository does not satisfy this contract. A correct initial inspector is expected to report `UPSTREAM_INVALID` because Phase 29 is incomplete.

## 10. Selective resume behavior

After a kernel restart, the Phase 30 notebook call must:

1. resolve the project root through the installed package;
2. inspect Phase 29 and Phase 30 canonical evidence;
3. validate every declared source path, checksum and run configuration;
4. classify Phase 30 state;
5. choose exactly one action;
6. render existing valid data, reconstruct derivatives, execute missing Phase 30 conditions only, wait, or block;
7. write or refresh the Phase 30 processing log only after its source evidence is validated;
8. return a compact persistent HTML result.

The notebook call must not depend on variables created by earlier notebook cells.

## 11. Output preservation contract

Before modifying the notebook, record:

- notebook checksum;
- cell count and stable cell identifiers;
- execution counts;
- per-cell output count;
- per-output MIME types;
- the exact cell and output containing the invalid widget view.

During refactoring:

- do not globally clear outputs;
- do not delete or replace unrelated outputs;
- replace only the invalid transient widget output with a persistent equivalent;
- allow a deliberately rerun Phase 30 cell to replace only its own output;
- compare the after-state with the preservation record.

## 12. Persistent presentation design

Replace the transient ipywidgets selector with static HTML and saved figures. Do not add `ipywidgets` as a dependency for this correction.

The Phase 30 display must contain only information needed to decide what happened:

- phase status and resolved action;
- prerequisite status;
- expected, verified and missing conditions;
- learning rate, run ID and validation metric for verified conditions;
- artifact revision and source-validation status;
- a concise block reason or next action when incomplete.

Full JSON remains in `docs/save_log_in_processing`; it is not dumped into the notebook.

## 13. Signed artifact revision policy

Do not overwrite the current invalid Phase 30 signed artifacts. Preserve them as historical evidence and produce an approved new revision when the prerequisite and scientific reruns are later authorized.

A new revision should use revisioned scientific artifacts such as:

- `results_v2.csv`;
- `sweep_manifest_v2.json`;
- `s8_learning_rate_winner_v2.json`;
- `s8_reference_update_v2.json`;
- `phase_30_signoff_v2.json`;
- an explicit active-revision pointer.

The processing log may be regenerated from the active validated revision because it is a derived presentation record.

## 14. Expected files in the later implementation

Subject to the architecture amendment and user approval, the implementation may modify:

- `COURSE_WORK/docs/RULE_BASE/architecture_rule.md`;
- `COURSE_WORK/src/course_work/experiments/phase_execution.py`;
- `COURSE_WORK/src/course_work/sweeps/sweep_results.py`;
- `COURSE_WORK/src/course_work/reporting/phase_summary.py`;
- `COURSE_WORK/scripts/run_single_condition.py`;
- `COURSE_WORK/scripts/run_phase_background.py`;
- `COURSE_WORK/scripts/run_phase_background/_notebook.py`;
- `COURSE_WORK/scripts/sweep_results_to_csv.py`;
- `COURSE_WORK/notebook_course_work/CourseWork.ipynb`;
- focused unit, integration, contract and regression tests.

The implementation must not modify raw data, model definitions, shared metrics, the Test set, Phase 0 to Phase 21 scientific outputs or unrelated notebook outputs.

## 15. Sequential implementation and verification plan

### Step 1. Capture the notebook preservation baseline

Create a machine-readable preservation manifest for cells and outputs without changing the notebook.

Immediate verification:

- notebook JSON parses;
- every cell ID is unique;
- the widget error output is identified exactly once;
- baseline checksum and output counts are reproducible.

### Step 2. Amend and approve the architecture contract

Add Phase 15 to Phase 30 ownership, the state and action models, revision policy and notebook boundary.

Immediate verification:

- no ownership conflicts remain;
- the new workflow respects the existing canonical-versus-derived hierarchy;
- implementation remains paused until explicit approval.

### Step 3. Implement the read-only phase-state inspector

Build the dependency and evidence inspector without training or artifact mutation.

Immediate verification:

- valid fixtures resolve to `VALID_REUSABLE`;
- a missing declared file invalidates PASS;
- a checksum mismatch invalidates PASS;
- a run learning-rate mismatch invalidates the condition;
- Phase 29 invalidity resolves Phase 30 to `UPSTREAM_INVALID`.

### Step 4. Harden signoff reuse

Require complete declared outputs, checksum matches, expected condition coverage and run lineage before returning an existing signoff.

Immediate verification:

- the current Phase 30 signoff is rejected as reusable;
- a complete fixture remains reusable;
- Test artifacts are never read.

### Step 5. Implement expected-minus-verified condition resolution

Derive missing work from the approved phase condition set and verified run registry rather than a hard-coded pending list.

Immediate verification:

- LR2 is recognized only when its Phase 29 reference is valid;
- LR1 and LR3 are independently classified;
- no verified condition is scheduled again;
- duplicate active runs resolve to `RUNNING`, not a second launch.

### Step 6. Remove hidden upstream materialization from the selective path

Refactor the condition and background launchers so Phase 30 missing-only execution does not call Phase 0 to Phase 29 materializers.

Immediate verification:

- call tracing shows no upstream materializer invocation;
- the selected phase and condition are explicit;
- invalid upstream or environment state blocks before process launch.

### Step 7. Quarantine non-canonical execution paths

Exclude environment-bypass, monkey-patch, synthetic-result and header-only result generation from the canonical dispatcher.

Immediate verification:

- canonical entry points cannot import or invoke those paths;
- empty scientific sources cannot produce a completed result artifact;
- existing historical files remain preserved.

### Step 8. Implement persistent compact Phase 30 rendering

Create widget-free HTML output from the validated inspector result and derived log.

Immediate verification:

- output MIME types are persistent;
- no widget-view MIME is emitted;
- tables use compact aligned columns;
- a blocked state is understandable without displaying raw JSON.

### Step 9. Reduce the Phase 30 notebook cell to one public call

Replace direct CSV reading and plotting with a single independently importable public API call.

Immediate verification:

- the cell runs after a clean kernel restart;
- it does not require variables from previous cells;
- it contains no data-processing or training logic;
- it does not call earlier phases.

### Step 10. Replace only the invalid final widget output

Replace the non-persistent saved widget view with its static persistent equivalent.

Immediate verification:

- the notebook contains no widget-view output;
- all unrelated cell sources and outputs match the preservation baseline;
- notebook JSON and notebook validation pass.

### Step 11. Run the Phase 30 recovery gate without training

Invoke the read-only inspector and render its decision.

Expected current result:

- Phase 30 reports `BLOCK`;
- the report identifies incomplete Phase 29 evidence;
- no phase materializer and no training process starts;
- Phase 0 to Phase 28 remain untouched.

If the result differs, stop and create a new issue and fix plan before continuing.

### Step 12. Run focused non-training tests

Add and run tests for state resolution, source validation, checksum enforcement, run lineage, dependency blocking, selective dispatch, persistent output and output preservation.

Immediate verification:

- all new tests pass;
- existing Phase 0 to Phase 14 tests still pass;
- no training process and no Test evaluation occurs.

### Step 13. Separate scientific recovery approval

After the workflow refactor passes, prepare a separate plan for correcting Phase 29 and then executing only missing Phase 30 conditions.

This step requires explicit approval because it can start training and create a new signed artifact revision.

### Step 14. Final regression and audit report

After any separately approved scientific recovery, verify active revisions, logs, notebook rendering and absence of unintended changes.

Immediate verification:

- Phase 30 can render from validated files after restart;
- deleting only the derived log triggers `REBUILD_LOG_ONLY`;
- removing one condition fixture triggers `EXECUTE_MISSING_ONLY` in controlled tests;
- stale PASS cannot bypass source validation;
- notebook preservation comparison passes.

## 16. Acceptance criteria

- The final notebook no longer contains an unreconstructable widget output.
- Phase 30 can be inspected from a clean kernel with one notebook call.
- Valid results are rendered without retraining.
- Missing derived logs are rebuilt without retraining.
- Only missing verified Phase 30 conditions can be scheduled.
- Invalid Phase 29 blocks Phase 30 without rerunning earlier phases.
- A PASS log or signoff with a missing source file is rejected.
- Run configurations must match their assigned learning-rate condition.
- No unrelated notebook output is removed or replaced.
- Test remains locked.
- No synthetic or bypass result is accepted as canonical evidence.

## 17. Rollback strategy

- Preserve the original notebook and its output manifest before notebook edits.
- Keep source changes in small, step-scoped patches.
- Do not overwrite signed scientific artifacts.
- If a verification fails, stop, record the failure in `analysis_error`, create a focused correction plan and restore only the affected step from its preservation baseline.

## 18. Approval state

- `USER_APPROVED_PREPROCESS_PLAN = true`
- `SOURCE_IMPLEMENTATION_ALLOWED = true`
- `NOTEBOOK_EDIT_ALLOWED = true`
- `SCIENTIFIC_EXECUTION_ALLOWED = false`

The user approved the workflow refactor on `2026-08-22`. Scientific recovery remains a separate approval and is not authorized by this plan.

## 19. Approved-plan execution record

### 19.1 Completed workflow refactor

- added a read-only Phase 23 to Phase 30 state inspector and expected-minus-verified condition resolver;
- hardened PASS signoff reuse with declared-path and checksum validation;
- removed upstream materialization from the selective execution entry points;
- blocked empty scientific sources from producing header-only result files;
- implemented compact static HTML for selective phase decisions and aggregate processing logs;
- reduced Phase 22 to Phase 30 notebook cells to source-owned public calls;
- replaced only the Phase 30 and final aggregate stored outputs;
- preserved the stored output hashes of every other notebook cell;
- removed the unreconstructable widget MIME from the notebook;
- produced a derived Phase 30 log with `UPSTREAM_INVALID` and `BLOCK` without training.

### 19.2 Plan adjustment applied during execution

Notebook JSON validation failed before output replacement because the final source array was not closed. The workflow stopped, repaired the JSON boundary and revalidated the full notebook before continuing.

Notebook boundary tests then identified direct data loading, plotting and comments in Phase 22 to Phase 29 cells. The plan was expanded to move those cells to the existing reporting and selective-resume public APIs while preserving all stored outputs.

### 19.3 Verification result

- focused selective execution and reporting tests: 26 passed;
- reporting, notebook-boundary and selective-policy tests: 37 passed;
- final combined workflow regression: 58 passed;
- notebook structure: 94 cells and 94 unique identifiers;
- widget MIME count: zero;
- non-target notebook output hash mismatches: zero;
- Python compilation: passed;
- patch whitespace validation: passed.

### 19.4 Scientific recovery remains blocked

The full suite reached 184 passing tests but retained 15 failures and 6 setup errors caused by the active no-GPU environment and invalid or missing scientific artifacts upstream of Phase 30. The approved workflow refactor does not authorize changing those artifacts or launching training.
