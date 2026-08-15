# PRE-PROCESS PLAN - REFACTOR PHASE 0 TO PHASE 5

## 1. Plan metadata

```text
Plan ID: CW-REFACTOR-0005-001
Issue ID: CW-PHASE-0005-001
Plan type: Sequential architecture and implementation refactor
Scope: Phase 0 through Phase 5
Primary notebook: COURSE_WORK/notebook_course_work/CourseWork.ipynb
Status: COMPLETED
Execution mode: STRICTLY_SEQUENTIAL
Parallel execution: FORBIDDEN
```

## 2. Objective

Refactor the current exploratory notebook into a reproducible Phase 0-5 workflow that satisfies every mandatory Phase contract and keeps all reusable processing logic outside the notebook.

The completed workflow must provide:

```text
Approved architecture ownership
Phase 0 coursework contract
Phase 1 reproducible environment contract
Phase 2 canonical data provenance
Phase 3 schema audit
Phase 4 temporal integrity audit
Phase 5 reproducible EDA
Machine-readable artifacts
Phase sign-offs
Automated validation
Orchestration-only notebook
Clean top-to-bottom execution
```

## 3. Confirmed issue source

This plan resolves:

```text
COURSE_WORK/docs/plan-doc/analysis_error/phase_0_to_5_coursework_nonconformance_issue.md
```

No implementation may begin if that issue record does not exist or its baseline hashes no longer match.

## 4. Governing contracts

Execution must follow:

```text
COURSE_WORK/working_rule.md
COURSE_WORK/docs/RULE_BASE/rule_code.md
COURSE_WORK/docs/RULE_BASE/architecture_rule.md after Step A1 approval and creation
COURSE_WORK/docs/plan-doc/plan_detail_for_each_phase/Phase_0_Coursework_contract.md
COURSE_WORK/docs/plan-doc/plan_detail_for_each_phase/Phase_1_Environment.md
COURSE_WORK/docs/plan-doc/plan_detail_for_each_phase/Phase_2_Data_acquisition.md
COURSE_WORK/docs/plan-doc/plan_detail_for_each_phase/Phase_3_Schema_audit.md
COURSE_WORK/docs/plan-doc/plan_detail_for_each_phase/Phase_4_Temporal_integrity_audit.md
COURSE_WORK/docs/plan-doc/plan_detail_for_each_phase/Phase_5_EDA.md
```

## 5. Current baseline

```text
Notebook SHA-1: 353e8ea2db2713041ae8aa34903907a40bce93bd
Raw CSV SHA-1: 346ce24141cff53b76bb74a87832c916bd23c860
Raw CSV SHA-256: 2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d
requirements.txt size: 0 bytes
architecture_rule.md: missing
artifacts directory: missing
Phase 0-5 sign-offs: missing
Current source implementation: empty scaffolds
```

## 6. Scope

### 6.1. In scope after approval

```text
Create the canonical architecture rule.
Correct .gitignore so source modules under src/course_work/data are trackable.
Create only the source packages required by Phase 0-5.
Implement reusable Phase 0-5 logic.
Create tests for each Phase contract.
Create deterministic artifacts and sign-offs.
Refactor CourseWork.ipynb into orchestration-only form.
Run clean sequential verification.
```

### 6.2. Out of scope

```text
Phase 6 or later implementation
Feature engineering for model inputs
Chronological Train/Validation/Test split
Scaling
Window creation
Model implementation
Training
Hyperparameter tuning
LoRA or fine-tuning execution
Final Test access
Attention extraction
Git commit or push
```

## 7. Proposed canonical architecture for Phase 0-5

```text
COURSE_WORK/
├── configs/
│   └── base/
│       └── coursework_contract.json
├── data/
│   └── raw_data/
│       ├── source/
│       │   └── appliances_energy_prediction.zip
│       ├── energydata_complete.csv
│       ├── checksums.sha256
│       ├── dataset_manifest.json
│       ├── source_metadata.json
│       ├── variable_metadata.csv
│       └── README_SOURCE.md
├── artifacts/
│   ├── contracts/
│   ├── environment/
│   ├── acquisition/
│   ├── schema/
│   ├── temporal/
│   └── eda/
│       ├── tables/
│       └── figures/
├── src/course_work/
│   ├── __init__.py
│   ├── contracts/
│   │   ├── __init__.py
│   │   └── coursework.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── acquisition.py
│   │   ├── schema.py
│   │   ├── temporal.py
│   │   └── eda.py
│   ├── reporting/
│   │   ├── __init__.py
│   │   └── eda.py
│   └── utils/
│       ├── __init__.py
│       ├── artifacts.py
│       ├── environment.py
│       └── reproducibility.py
├── tests/
│   ├── contracts/
│   ├── unit/
│   └── integration/
└── notebook_course_work/
    └── CourseWork.ipynb
```

Only directories and files required for Phase 0-5 may be created in this plan.

## 8. Ownership rules

### 8.1. Notebook

The notebook may contain only:

```text
Markdown narrative
Imports from course_work
Configuration reference selection
Calls to public Phase APIs
Display of returned summaries, tables and figures
Display of sign-off state
```

The notebook must not contain:

```text
Function or class definitions
Data transformation logic
Checksum implementation
Schema implementation
Timestamp implementation
EDA computation
Plot-construction logic
Artifact-writing logic
Outlier removal
Interpolation
```

### 8.2. Contracts

`contracts/coursework.py` owns validation and materialization of the Phase 0 contract.

`configs/base/coursework_contract.json` owns the approved machine-readable values.

### 8.3. Environment

`utils/environment.py` owns interpreter, package, platform and device inventory.

`utils/reproducibility.py` owns seed configuration and reproducibility smoke tests.

### 8.4. Data

`data/acquisition.py` owns Phase 2 provenance and integrity verification.

`data/schema.py` owns Phase 3 schema audit.

`data/temporal.py` owns Phase 4 temporal audit and continuity segmentation.

`data/eda.py` owns Phase 5 numerical EDA tables and hypotheses.

### 8.5. Reporting

`reporting/eda.py` owns Phase 5 figure creation and export from validated EDA inputs.

### 8.6. Artifact utilities

`utils/artifacts.py` owns safe path resolution, atomic writes, checksums and machine-readable serialization.

It must not own scientific logic.

## 9. Allowed dependency direction

```text
notebook
-> contracts, data, reporting, utils public APIs

reporting
-> validated data summaries

data
-> contracts and utils

contracts
-> standard library and schema validation helpers

utils
-> standard library and approved runtime dependencies
```

Forbidden:

```text
src -> notebook
contracts -> data
data -> reporting
raw data -> generated output
Phase N -> unsigned Phase N-1 output
```

## 10. Files allowed to change after approval

Existing files:

```text
.gitignore
COURSE_WORK/README.md
COURSE_WORK/requirements.txt
COURSE_WORK/notebook_course_work/CourseWork.ipynb
COURSE_WORK/src/course_work/data/acquisition.py
COURSE_WORK/src/course_work/data/schema.py
COURSE_WORK/src/course_work/data/temporal.py
```

New governance and configuration files:

```text
COURSE_WORK/docs/RULE_BASE/architecture_rule.md
COURSE_WORK/configs/base/coursework_contract.json
```

New source files:

```text
COURSE_WORK/src/course_work/__init__.py
COURSE_WORK/src/course_work/contracts/__init__.py
COURSE_WORK/src/course_work/contracts/coursework.py
COURSE_WORK/src/course_work/data/__init__.py
COURSE_WORK/src/course_work/data/eda.py
COURSE_WORK/src/course_work/reporting/__init__.py
COURSE_WORK/src/course_work/reporting/eda.py
COURSE_WORK/src/course_work/utils/__init__.py
COURSE_WORK/src/course_work/utils/artifacts.py
COURSE_WORK/src/course_work/utils/environment.py
COURSE_WORK/src/course_work/utils/reproducibility.py
```

New test files:

```text
COURSE_WORK/tests/contracts/test_coursework_contract.py
COURSE_WORK/tests/unit/test_environment.py
COURSE_WORK/tests/unit/test_reproducibility.py
COURSE_WORK/tests/unit/test_acquisition.py
COURSE_WORK/tests/unit/test_schema.py
COURSE_WORK/tests/unit/test_temporal.py
COURSE_WORK/tests/unit/test_eda.py
COURSE_WORK/tests/integration/test_phase_0_to_5_chain.py
COURSE_WORK/tests/integration/test_notebook_boundary.py
```

Runtime outputs allowed after their Phase gate passes:

```text
COURSE_WORK/artifacts/contracts/*
COURSE_WORK/artifacts/environment/*
COURSE_WORK/artifacts/acquisition/*
COURSE_WORK/artifacts/schema/*
COURSE_WORK/artifacts/temporal/*
COURSE_WORK/artifacts/eda/*
COURSE_WORK/data/raw_data/source/appliances_energy_prediction.zip
COURSE_WORK/data/raw_data/checksums.sha256
COURSE_WORK/data/raw_data/dataset_manifest.json
COURSE_WORK/data/raw_data/source_metadata.json
COURSE_WORK/data/raw_data/variable_metadata.csv
COURSE_WORK/data/raw_data/README_SOURCE.md
```

## 11. Files explicitly protected

```text
COURSE_WORK/data/raw_data/energydata_complete.csv
COURSE_WORK/working_rule.md
COURSE_WORK/docs/RULE_BASE/rule_code.md
COURSE_WORK/docs/plan-doc/plan_detail_for_each_phase/*
COURSE_WORK/docs/plan-doc/plan_overview/*
COURSE_WORK/docs/plan-doc/analysis_error/phase_0_to_5_coursework_nonconformance_issue.md
COURSE_WORK/docs/plan-doc/plan_before_process/refactor_course_work_architecture_scaffold_plan.md
Every Phase 6-59 implementation file
```

The raw CSV hash must be checked after every Phase that reads data.

## 12. `.gitignore` correction

Current pattern:

```text
data/
```

incorrectly ignores:

```text
COURSE_WORK/src/course_work/data/
```

The approved correction will scope data ignores to data-storage roots while keeping Python source trackable.

Proposed behavior:

```text
Repository data assets remain ignored according to policy.
COURSE_WORK/src/course_work/data/*.py is not ignored.
Runtime artifacts follow an explicit policy.
```

Validation must use `git check-ignore` before proceeding.

## 13. Coding conventions

```text
Python files use snake_case.
Classes use PascalCase.
Constants use UPPER_SNAKE_CASE.
Public functions have explicit typed inputs and outputs.
Paths are pathlib.Path values resolved from project root.
No absolute machine-specific path is stored.
No comments or icons are added to code.
No silent exception handling.
No fabricated values.
No inplace mutation of canonical DataFrames.
No unapproved dependency.
No Test access.
```

Scientific outputs must retain full precision in machine-readable files. Display formatting must not modify stored values.

## 14. Sequential execution workflow

### Step A0 - Revalidate baseline

Actions:

```text
Recompute notebook SHA-1.
Recompute raw CSV SHA-1 and SHA-256.
Recheck Git state.
Recheck required rules and Phase files.
Recheck that Phase 6 has not started.
```

Validation:

```text
Notebook matches approved baseline or change is explained.
Raw CSV SHA-256 equals 2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d.
Issue record exists.
This plan is approved.
```

Stop if any baseline changed unexpectedly.

### Step A1 - Create and approve architecture rule

Create:

```text
COURSE_WORK/docs/RULE_BASE/architecture_rule.md
```

It must define:

```text
Canonical root
Directory ownership
Source ownership
Dependency direction
Notebook orchestration-only boundary
Data lifecycle
Artifact lifecycle
Test ownership
Phase-to-module mapping
Change-control rules
```

Immediate validation:

```text
The file exists and is substantive.
All Phase 0-5 owner modules are mapped.
No existing Phase contract is weakened.
No conflict with working_rule.md or rule_code.md.
```

Gate:

```text
Architecture implementation remains blocked until Human approves the written architecture rule.
```

### Step A2 - Correct source tracking and create minimal packages

Actions:

```text
Correct .gitignore data-path scope.
Create package directories and __init__.py files listed in Section 10.
Create artifact directories only when their producing Phase executes.
```

Immediate validation:

```text
git check-ignore confirms source data modules are trackable.
Python package imports resolve.
No Phase logic exists in notebook.
No unrelated path is created.
```

Regression link:

```text
Revalidate Step A1 architecture mapping.
Revalidate raw CSV hash.
```

### Step P0 - Implement Phase 0

Implement:

```text
Task: multivariate_time_series_regression
Dataset: UCI Appliances Energy Prediction
Target: Appliances
Target unit: Wh
Forecast horizon: one 10-minute step
Lookbacks: 36, 72, 144
Primary lookback: 144
Feature sets: FS0, FS1, FS2
Time features: TF0, TF1
Target scaling: YS0, YS1
Boundary protocols: WB0, WB1
Models: Persistence, LSTM, Transformer Encoder
Selection metric: Validation RMSE
Final metrics: MAE, RMSE, R2
Chronological split: 70, 15, 15
Final seeds: 42, 123, 2026
RQ1-RQ7
Fairness, leakage and attention rules
Protocol violations
Definition of Done
```

Outputs:

```text
configs/base/coursework_contract.json
artifacts/contracts/coursework_contract.json
artifacts/contracts/coursework_contract.sha256
artifacts/contracts/phase_0_signoff.json
```

Immediate validation:

```text
Contract schema test passes.
Option IDs are complete.
Percentages sum to 1.0.
Horizon and target are unambiguous.
No data-dependent computation occurs.
Phase 0 sign-off is PASS.
```

Regression link:

```text
Architecture rule still passes.
Raw CSV hash is unchanged.
```

Stop before Phase 1 if Phase 0 does not pass.

### Step P1 - Implement Phase 1

Implement:

```text
Minimal approved dependencies
Interpreter and kernel checks
Python and package versions
Platform and working-directory inventory
CUDA, MPS and CPU detection
Automatic device selection
Float dtype contract
set_seed(seed)
D0 practical reproducibility
D1 strict diagnostic mode
Tensor smoke test
Autograd smoke test
nn.Module and optimizer-step smoke test
Python, NumPy and PyTorch randomness smoke tests
```

Outputs:

```text
artifacts/environment/environment_report.json
artifacts/environment/requirements_freeze.txt
artifacts/environment/smoke_test_report.json
artifacts/environment/phase_1_signoff.json
```

Immediate validation:

```text
Phase 0 sign-off and contract checksum verify.
Core imports pass.
Interpreter and kernel contract pass.
Selected device executes forward and backward.
Loss and gradients are finite.
Seed repeatability smoke test passes.
Environment report contains no fabricated field.
Phase 1 sign-off is PASS.
```

Regression link:

```text
Revalidate Phase 0 contract checksum.
Revalidate architecture imports.
Revalidate raw CSV hash.
```

Stop before Phase 2 if Phase 1 does not pass.

### Step P2 - Implement Phase 2

Selected acquisition method:

```text
AQ0 official UCI archive download
```

Network action requires explicit runtime permission if sandboxed.

Actions:

```text
Download official archive to a temporary path.
Verify ZIP integrity.
Verify expected CSV member.
Compute archive SHA-256.
Safely extract candidate CSV to a temporary path.
Compute candidate CSV SHA-256.
Compare candidate CSV with the existing canonical raw CSV.
Do not overwrite the existing raw CSV.
Freeze the archive only after integrity checks pass.
Record source, DOI, license and acquisition timestamps.
Optionally cross-check ucimlrepo metadata without using it as raw source.
```

Mandatory mismatch behavior:

```text
If the official extracted CSV hash differs from the existing raw CSV hash, stop.
Do not replace either file.
Create a discrepancy report and wait for Human decision.
```

Outputs:

```text
data/raw_data/source/appliances_energy_prediction.zip
data/raw_data/checksums.sha256
data/raw_data/dataset_manifest.json
data/raw_data/source_metadata.json
data/raw_data/variable_metadata.csv when available
data/raw_data/README_SOURCE.md
artifacts/acquisition/acquisition_log.json
artifacts/acquisition/phase_2_signoff.json
```

Immediate validation:

```text
Phase 0 and Phase 1 sign-offs verify.
Official source identity is UCI ID 374.
DOI and CC BY 4.0 are recorded.
ZIP integrity passes.
CSV checksum matches the protected baseline.
CSV minimal parse passes.
Raw file content is unchanged.
No date conversion, split, scaling or feature removal occurs.
DATA-v1 is assigned.
Phase 2 sign-off is PASS.
```

Regression link:

```text
Revalidate Phase 0 contract.
Revalidate ENV-v1.
Revalidate raw CSV hash.
```

Stop before Phase 3 if Phase 2 does not pass.

### Step P3 - Implement Phase 3

Actions:

```text
Verify DATA-v1 checksum.
Load canonical raw CSV without dtype overrides.
Record exact shape and ordered columns.
Compare expected and actual schema.
Audit duplicate names, whitespace and case.
Record raw dtypes.
Assign semantic roles, groups, units and descriptions.
Audit null, all-null, constant and near-constant columns.
Audit numeric coercion and non-finite values.
Perform timestamp parse probe without mutating raw DataFrame.
Classify rv1 and rv2 as random controls.
Reconcile 28 predictors with 29 raw columns.
Create schema fingerprint.
```

Outputs:

```text
artifacts/schema/schema_summary.csv
artifacts/schema/variable_dictionary.csv
artifacts/schema/schema_comparison.csv
artifacts/schema/schema_manifest.json
artifacts/schema/schema_fingerprint.txt
artifacts/schema/schema_discrepancies.json
artifacts/schema/phase_3_signoff.json
```

Immediate validation:

```text
Phase 2 sign-off and DATA-v1 checksum verify.
Rows equal 19,735.
Columns equal 29.
Ordered raw columns match the canonical CSV.
Target exists exactly once.
Timestamp exists exactly once.
Timestamp probe succeeds using %Y-%m-%d %H:%M:%S.
Raw DataFrame remains unchanged.
SCHEMA-v1 is assigned.
Phase 3 sign-off is PASS or documented PASS_WITH_WARNING.
```

Regression link:

```text
Revalidate Phase 0-2 sign-offs.
Revalidate raw CSV hash.
Revalidate no acquisition artifact changed.
```

Stop before Phase 4 if Phase 3 fails.

### Step P4 - Implement Phase 4

Actions:

```text
Verify DATA-v1 and SCHEMA-v1.
Load raw data and create a deep derived copy.
Strictly parse timestamps.
Audit original order before sorting.
Audit negative deltas and zero deltas.
Audit exact and conflicting duplicate timestamps.
Create a stable sorted derived view.
Compute interval distribution.
Audit 10-minute grid alignment.
Build expected timestamp grid.
List missing timestamps and gaps.
Compute coverage and continuity metrics.
Audit hourly and daily counts.
Create continuity segments.
Create window-safety contract.
Do not interpolate or resample.
```

Outputs:

```text
artifacts/temporal/temporal_manifest.json
artifacts/temporal/temporal_summary.csv
artifacts/temporal/timestamp_gaps.csv
artifacts/temporal/missing_timestamps.csv
artifacts/temporal/duplicate_timestamps.csv
artifacts/temporal/continuity_segments.csv
artifacts/temporal/interval_distribution.csv
artifacts/temporal/daily_observation_counts.csv
artifacts/temporal/temporal_discrepancies.json
artifacts/temporal/phase_4_signoff.json
```

Immediate validation:

```text
Phase 2 and Phase 3 artifacts verify.
Parse failure count is zero or explicitly blocks the Phase.
Original-order status is recorded before sort.
Every interval is classified.
Duplicate and gap status are known.
Continuity segments cover all valid rows without overlap.
Window-safety contract rejects gap crossing.
Raw DataFrame and raw CSV remain unchanged.
TEMPORAL-v1 is assigned.
Phase 4 sign-off is PASS or documented PASS_WITH_WARNING.
```

Regression link:

```text
Revalidate Phase 0-3 sign-offs.
Revalidate DATA-v1 and SCHEMA-v1 checksums.
Revalidate raw CSV hash.
```

Stop before Phase 5 if Phase 4 fails.

### Step P5 - Implement Phase 5

Actions:

```text
Verify DATA-v1, SCHEMA-v1 and TEMPORAL-v1.
Create df_eda from the validated derived temporal view.
Preserve raw_row_index and continuity_segment_id.
Create EDA-only temporal columns in df_eda.
Compute target summary and quantiles.
Create deterministic 24-hour and 7-day windows.
Describe target spikes without removing them.
Compute hourly and weekday mean and median profiles.
Compute hour-by-weekday profile.
Summarize temperature, humidity, weather, lights, rv1 and rv2.
Compute correlation matrix and target ranking.
Identify high-correlation feature pairs.
Compute selected-lag target autocorrelation.
Compute segment-aware exogenous cross-correlation when justified.
Compute segment-aware rolling mean and standard deviation.
Export extreme-target samples.
Create hypothesis registry.
Create anomaly log.
Render and save the mandatory figure set.
Do not clean, interpolate, select features or tune models.
```

Outputs:

```text
artifacts/eda/eda_manifest.json
artifacts/eda/tables/eda_numeric_summary.csv
artifacts/eda/tables/target_quantiles.csv
artifacts/eda/tables/hourly_energy_profile.csv
artifacts/eda/tables/weekday_energy_profile.csv
artifacts/eda/tables/correlation_matrix.csv
artifacts/eda/tables/target_correlations.csv
artifacts/eda/tables/high_correlation_pairs.csv
artifacts/eda/tables/selected_lag_correlations.csv
artifacts/eda/tables/extreme_target_samples.csv
artifacts/eda/tables/eda_hypotheses.csv
artifacts/eda/eda_anomalies.json
artifacts/eda/figures/EDA_01_target_histogram.png through required EDA figures
artifacts/eda/phase_5_signoff.json
```

Immediate validation:

```text
Phase 2-4 input versions and checksums verify.
df_raw is unchanged.
df_eda carries row and continuity lineage.
All mandatory tables exist with non-empty schemas.
All mandatory figures exist with canonical names.
Lag and rolling analysis respect continuity segments.
No outlier removal or interpolation occurs.
No final feature, lookback, loss or model is selected.
Every decision-sensitive observation is labeled as a hypothesis.
EDA-v1 is assigned.
Phase 5 sign-off is PASS.
```

Regression link:

```text
Revalidate Phase 0-4 sign-offs.
Revalidate all upstream checksums.
Revalidate raw CSV hash.
```

Stop before notebook reconstruction if Phase 5 fails.

### Step N1 - Rebuild notebook as orchestration-only

The notebook will contain six clearly labeled Phase sections.

Each section will contain:

```text
Phase objective Markdown
Contract summary Markdown
One or more short public-API calls
Display of validated outputs
Display of sign-off state
```

The notebook must not contain:

```text
def
class
pd.read_csv
fetch_ucirepo
pd.to_datetime
DataFrame mutation
correlation implementation
rolling implementation
interpolation
plot-construction implementation
artifact serialization
```

Immediate validation:

```text
Notebook JSON is valid.
Notebook boundary test passes.
All imported APIs exist.
All Phase sections are present in order.
No stale outputs remain before clean execution.
```

Regression link:

```text
Revalidate every Phase source test.
Revalidate every Phase artifact and sign-off.
Revalidate raw CSV hash.
```

### Step N2 - Clean top-to-bottom execution

Actions:

```text
Use the verified Phase 1 kernel.
Clear stale outputs and execution counts.
Execute from the first cell to the final cell exactly once.
Capture the resulting notebook state.
```

Final validation:

```text
All code cells have execution counts.
Execution counts are monotonic.
No error output exists.
No unresolved warning exists.
No network fetch occurs in the notebook.
No raw mutation occurs.
All Phase 0-5 sign-offs are PASS or approved documented warning where allowed.
Raw SHA-256 is unchanged.
Integration chain test passes.
```

## 15. Verification after every step

Every execution report must include:

```text
Step ID
Files changed
Input hashes
Output hashes
Tests run
Validation result
Upstream regression result
Current-step result
Downstream compatibility result
Unexpected findings
Next gate
```

No step may be reported as complete using visual inspection alone.

## 16. Test strategy

### 16.1. Contract tests

```text
Phase 0 schema and option completeness
Phase-gate dependency validation
Artifact field contracts
Sign-off status contracts
```

### 16.2. Unit tests

```text
Atomic artifact writes
Hash reproducibility
Environment inventory
Seed reproducibility
Archive integrity helpers
Schema comparison
Semantic role mapping
Strict timestamp parsing
Original-order detection
Duplicate classification
Grid and gap detection
Continuity segmentation
Segment-aware lag calculations
EDA summary schemas
```

### 16.3. Integration tests

```text
Phase 0 -> Phase 1 gate
Phase 1 -> Phase 2 gate
Phase 2 -> Phase 3 gate
Phase 3 -> Phase 4 gate
Phase 4 -> Phase 5 gate
Raw checksum preservation
Notebook orchestration boundary
```

## 17. Risk register

### Risk 1 - Architecture contract is missing

Mitigation:

```text
Step A1 must complete and receive separate Human approval before source implementation.
```

### Risk 2 - Official archive differs from current raw CSV

Mitigation:

```text
Compare in temporary storage.
Never overwrite either artifact.
Stop and create a discrepancy plan.
```

### Risk 3 - Current environment lacks dependencies

Mitigation:

```text
Audit installed versions first.
Request permission before any network installation.
Do not silently upgrade packages.
```

### Risk 4 - Notebook output is large

Mitigation:

```text
Store canonical figures as artifacts.
Display report-sized outputs only.
Avoid redundant embedded images.
```

### Risk 5 - EDA accidentally becomes preprocessing

Mitigation:

```text
Use immutable raw input and a dedicated df_eda.
Ban interpolation and deletion in Phase 5 tests.
```

### Risk 6 - `.gitignore` hides source files

Mitigation:

```text
Correct path scope and validate with git check-ignore before implementation.
```

### Risk 7 - Platform-dependent results

Mitigation:

```text
Record environment and device.
Use deterministic windows and seeded operations.
Keep numerical tolerances explicit.
```

## 18. Stop conditions

Stop immediately when:

```text
Raw CSV hash changes.
An upstream sign-off fails.
Architecture rule conflicts with a Phase contract.
Official archive integrity fails.
Official CSV differs from the protected raw CSV.
A required dependency is unavailable.
A test fails without an approved explanation.
Notebook requires processing logic to pass.
An unplanned file or dependency is required.
Phase scope expands beyond Phase 5.
Test data or final Test protocol is accessed.
```

On stop:

```text
Do not continue.
Preserve evidence.
Create an issue update.
Create a correction plan.
Wait for Human approval.
```

## 19. Rollback policy

```text
Do not use destructive reset.
Do not overwrite the raw CSV.
Preserve the audited notebook until the rebuilt notebook is validated.
Use file-level restoration only after Human approval.
Keep failed artifacts versioned or isolated for diagnosis.
Do not silently rewrite a failed sign-off to PASS.
```

## 20. Expected final state

```text
Canonical architecture rule exists and is approved.
Source ownership is explicit.
Source data modules are trackable by Git.
Phase 0-5 reusable logic exists outside the notebook.
Every Phase has complete machine-readable artifacts.
Every Phase has a verified sign-off.
Notebook is orchestration-only.
Notebook executes cleanly from top to bottom.
Raw data hash is unchanged.
Phase 6 remains unstarted.
```

## 21. Definition of Done

The refactor is complete only when:

```text
Architecture validation passes.
All planned source and test files exist.
All contract tests pass.
All unit tests pass.
All integration tests pass.
Phase 0 PASS.
Phase 1 PASS.
Phase 2 PASS.
Phase 3 PASS or approved PASS_WITH_WARNING.
Phase 4 PASS or approved PASS_WITH_WARNING.
Phase 5 PASS.
Notebook boundary test passes.
Clean notebook execution passes.
Raw SHA-256 remains unchanged.
No Phase 6 work has started.
Final execution report is complete.
```

## 22. Approval gates

Gate 1:

```text
Human approves this complete plan.
```

Gate 2:

```text
Human approves the written architecture_rule.md after Step A1.
```

Gate 3:

```text
Human approves any dependency installation requiring network access.
```

Gate 4:

```text
Human resolves any official-data checksum mismatch.
```

No gate may be inferred from silence.

## 23. Current approval state

```text
ISSUE_CONFIRMED=true
PLAN_CREATED=true
PLAN_VALIDATED=true
HUMAN_APPROVED_PLAN=true
ARCHITECTURE_RULE_CREATED=true
ARCHITECTURE_RULE_APPROVED=true
IMPLEMENTATION_ALLOWED=true
PHASE_0_TO_5_COMPLETED=true
NOTEBOOK_ORCHESTRATION_ONLY=true
CLEAN_EXECUTION_PASS=true
FULL_TEST_SUITE_PASS=true
PHASE_6_STARTED=false
```

## 24. Completion result

```text
Architecture gates A0-A2: PASS
Phase gates P0-P5: PASS with documented SCHEMA-v1 warning SD-001
Notebook gate N1: PASS
Clean execution gate N2: PASS
Automated validation: 42/42 PASS
Raw data immutable: PASS
Signed output checksums: PASS
Phase 6 boundary: PRESERVED
```
