# Notebook Widget And Phase Selective Resume Issue

## 1. Issue identity

- Issue ID: `CW-NOTEBOOK-SELECTIVE-RESUME-001`
- Scope: `CourseWork.ipynb`, Phase 22 to Phase 30 execution, processing logs, sweep artifacts and notebook presentation
- Current disposition: investigation complete, correction not started

## 2. Requested behavior

The notebook must support restarting the kernel and running a selected late phase without rerunning every earlier phase. Before doing work, the selected phase must inspect its processing log and canonical artifacts, determine what is complete or missing, then choose one of these behaviors:

- render valid existing results only;
- rebuild a missing or stale presentation log without retraining;
- execute only missing conditions of the selected phase;
- block with a precise reason when a prerequisite is invalid.

Existing notebook outputs must not be globally cleared or deleted.

## 3. Inspection coverage

The inspection covered:

- 83 Python files and one notebook;
- all 94 notebook cells, including 47 code cells;
- all 31 processing log JSON files;
- Phase 22 to Phase 30 modules, scripts, manifests, signoffs and run records;
- notebook-boundary, reporting and phase tests;
- the architecture and process rules governing canonical and derived artifacts.

All 83 Python files parse successfully. All 31 processing logs are valid JSON. The notebook contains no stored Python exception output, but that does not establish that its saved results remain reproducible or complete.

## 4. Confirmed observations

### 4.1 Final notebook output is a non-persistent widget view

The final notebook cell stores an `application/vnd.jupyter.widget-view+json` output whose model ID is `c3e7900f8da74042a7238652c6733eb0`. The notebook does not contain the corresponding `metadata.widgets` model state. After the kernel closes, VS Code cannot reconstruct that model and reports that it cannot render the content.

Installing `ipywidgets` alone cannot restore widget state that was not saved. The final cell also mixes log rendering, HTML and JavaScript, training-history discovery, plotting and transient widgets in one notebook cell.

### 4.2 Phase 22 to Phase 30 are imported but not executed through their public phase APIs

The notebook imports `materialize_phase_22` through `materialize_phase_30`, but the Phase 23 to Phase 30 notebook cells directly read CSV files and perform plotting. They depend on in-memory variables established by earlier notebook cells, including `PROJECT_ROOT`, `pd`, `Path`, `display` and `render_dataframe_table`.

Consequently, Phase 30 is not independently runnable after a clean kernel restart.

### 4.3 Phase 23 to Phase 30 cells contain a missing-file control-flow defect

The DataFrame for each sweep is created only when its result CSV exists, while plotting code is placed in the missing-file branch and still references that DataFrame. All eight corresponding `results.csv` files are currently absent. A clean execution can therefore fail with an undefined variable even though the notebook currently stores older successful-looking outputs.

### 4.4 Processing logs are being treated as authoritative when they are derived records

The architecture rule defines processing logs as derived presentation records. Canonical evidence remains the phase manifest, signoff, run registry, run configuration, result artifacts and checksums. Current summary rendering reads the log status without revalidating those sources.

A JSON log that says `PASS` is therefore insufficient evidence for reuse.

### 4.5 Phase 30 reports PASS while a declared source artifact is missing

`phase_30_s8_learning_rate_log.json` and its signoff declare:

- `artifacts/sweeps/s8_learning_rate/sweep_manifest.json`, present and checksum-matching;
- `artifacts/sweeps/s8_learning_rate/results.csv`, missing;
- `artifacts/sweeps/s8_learning_rate/phase_30_signoff.json`, present and checksum-matching.

The Phase 30 materializer returns an existing PASS signoff without verifying all declared output paths and checksums. This allows a stale signoff and log to be reused.

### 4.6 Phase 30 run lineage is inconsistent

The Phase 30 records identify LR3, learning rate `0.001`, as the winning condition and associate it with run `RUN_TR_B0_0010_1CEB611E`. That run's canonical configuration records `training.learning_rate = 0.0003`. The same run therefore cannot serve as evidence for the claimed LR3 condition.

The expected Phase 30 outputs `s8_learning_rate_winner.json`, `s8_reference_update.json` and `results.csv` are missing. No valid fresh LR1 or LR3 run evidence was found.

### 4.7 Phase 30 manifest conflicts with its approved phase specification

The Phase 30 specification requires exactly:

- LR1: `0.0001`;
- LR2: `0.0003`, reused from Phase 29;
- LR3: `0.001`;
- validation RMSE in Wh as the selection metric;
- Test remaining locked.

The current manifest describes four different variants, including `0.0005` and `0.005`. It does not implement the approved Phase 30 comparison contract.

### 4.8 Phase 29 is not a valid prerequisite for Phase 30

The Phase 29 winner and reference-update artifacts are missing. Its current signoff represents only one batch-size condition even though the phase specification requires a B32 versus B64 comparison. Phase 30 must not silently train or claim completion while this dependency remains invalid.

### 4.9 Existing scripts do not implement selective resume

- `run_single_condition.py` calls a Phase 0 to Phase 13 materialization chain for every condition.
- `run_all_pending.py` uses a hard-coded pending list instead of deriving work from verified artifacts and run records.
- `run_phase_background.py` cannot request Phase 30 missing conditions only.
- `sweep_results_to_csv.py` can create a header-only CSV when source JSONL is absent, which can mask missing scientific results.

### 4.10 Bypass and synthetic-result scripts cannot be authoritative

Some scripts bypass the signed environment, monkey-patch environment checks, fabricate plausible sweep results or clear notebook outputs. These paths conflict with reproducibility, artifact lineage and the user's output-preservation requirement. They must not participate in the canonical selective-resume workflow.

### 4.11 Existing aggregate sweep status contradicts PASS signoffs

`artifacts/sweeps/all_sweep_results.json` reports no completed conditions and records environment-signature failures. This contradicts the existing PASS sweep signoffs and confirms that status cannot be accepted without cross-artifact validation.

### 4.12 Test coverage stops before the affected workflow

Current notebook-boundary and phase-summary coverage primarily stops at Phase 14. There are no focused tests for Phase 22 to Phase 30 artifact completeness, selective resume, condition-level dispatch, dependency blocking or persistent widget-free outputs.

### 4.13 Architecture documentation is behind the implementation

The architecture rule still describes Phase 0 to Phase 14 as the current implementation boundary, while the repository and notebook now contain later phases. Phase 15 to Phase 30 ownership, selective execution, log derivation, revision handling and notebook presentation must be approved in the architecture rule before source implementation proceeds.

## 5. Root causes

- RC1: a transient widget view was saved without its model state.
- RC2: late notebook cells depend on earlier in-memory notebook state.
- RC3: notebook cells duplicate processing instead of calling a single source-owned public API.
- RC4: PASS reuse does not validate declared files, checksums and run lineage.
- RC5: processing logs are trusted above canonical scientific evidence.
- RC6: there is no phase-state model distinguishing reusable, stale, incomplete and blocked states.
- RC7: pending work is hard-coded rather than computed as expected conditions minus verified completed conditions.
- RC8: Phase 30 artifacts and run attribution conflict with the approved phase contract.
- RC9: Phase 29 is incomplete, so Phase 30 currently has an invalid prerequisite.
- RC10: bypass and synthetic-generation utilities contaminate the available execution paths.
- RC11: test and architecture coverage have not advanced with Phase 22 to Phase 30.

## 6. Required correction direction

The correction must introduce a source-owned selective phase execution layer. It must validate dependencies and canonical evidence without invoking earlier materializers, classify current state, choose the smallest safe action and return a persistent notebook-ready presentation.

The required action set is:

- `RENDER_ONLY`;
- `REBUILD_LOG_ONLY`;
- `REBUILD_DERIVED_ONLY`;
- `EXECUTE_MISSING_ONLY`;
- `WAIT_FOR_RUNNING_PROCESS`;
- `BLOCK`.

The notebook must use a single public call for Phase 30. The final transient widget must be replaced by static persistent HTML and saved figures. Existing unrelated cell outputs must remain unchanged.

Because the current Phase 29 prerequisite is invalid, the expected immediate result of a correct Phase 30 inspection is `BLOCK`, with a precise dependency report. It must not rerun Phase 0 to Phase 29 and must not start Phase 30 training.

## 7. Must-not-change constraints

- Do not clear all notebook outputs.
- Do not rerun the whole notebook as part of diagnosis or correction.
- Do not treat a processing log alone as proof of scientific completion.
- Do not overwrite signed invalid historical artifacts in place.
- Do not unlock or inspect Test outcomes before their approved phase.
- Do not change the user's phase sequence or model protocol.
- Do not use bypass or synthetic-result scripts as canonical evidence.
- Do not implement source changes until the pre-process plan is approved.

## 8. Resolution gate

This issue is ready for planning. Source implementation, notebook edits, output replacement and scientific execution remain blocked until the associated pre-process plan and architecture amendment are approved.

## 9. Implementation audit findings

### 9.1 Notebook JSON boundary failure

The final cell source array was missing its closing JSON bracket after the approved source-only edit. The notebook could not be parsed at line 8614. The structure was repaired before any output replacement, then verified as 94 valid cells with 94 unique identifiers.

### 9.2 Phase 22 to Phase 29 notebook boundary violations

The newly added Phase 22 to Phase 29 cells contained direct CSV loading, conditional control flow and Matplotlib processing. Phase 23 to Phase 29 also placed plotting under the missing-file branch while referencing a dataframe that would not exist in that branch. These cells were reduced to source-owned reporting calls while their stored outputs were preserved.

### 9.3 Full-suite scientific-state blockers

The full non-training test invocation completed with 184 passing tests, 15 failures and 6 setup errors. The remaining failures group into existing scientific-state dependencies:

- Phase 1 rejects materialization because the active environment exposes neither CUDA nor MPS;
- processing logs reference missing learning-diagnostic and sweep result CSV files;
- experiment registry validation rejects current artifact and checksum lineage;
- downstream persistence and chronological-split tests invoke the invalid upstream materialization chain.

These failures cannot be corrected safely by weakening validation, fabricating files or starting scientific execution under the current approval state.

### 9.4 Resolution boundary

The notebook and selective-resume refactor is verified independently. Repairing the Phase 1 environment contract, registry lineage, missing Phase 22 to Phase 29 artifacts and scientific checksums requires a separate approved recovery plan before Phase 30 can execute missing conditions.
