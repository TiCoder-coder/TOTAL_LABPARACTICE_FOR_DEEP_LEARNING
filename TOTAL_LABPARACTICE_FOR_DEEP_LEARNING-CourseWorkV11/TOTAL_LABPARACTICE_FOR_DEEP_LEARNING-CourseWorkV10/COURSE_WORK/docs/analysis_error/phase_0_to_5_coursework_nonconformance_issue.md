# ISSUE SYNTHESIS - PHASE 0 TO PHASE 5 COURSEWORK NONCONFORMANCE

## 1. Issue metadata

```text
Issue ID: CW-PHASE-0005-001
Issue type: Architecture, reproducibility, protocol and Phase-gate nonconformance
Scope: Phase 0 through Phase 5
Primary target: COURSE_WORK/notebook_course_work/CourseWork.ipynb
Audit mode: Read-only static audit
Status: RESOLVED
Severity: CRITICAL
Execution state: COMPLETED
```

## 2. Objective of this issue record

Record the confirmed reasons why the current notebook cannot be accepted as a completed implementation of Phase 0 through Phase 5.

This record separates observed evidence from the future correction workflow. It does not implement a fix.

## 3. Sources audited

```text
COURSE_WORK/working_rule.md
COURSE_WORK/docs/RULE_BASE/rule_code.md
COURSE_WORK/docs/plan-doc/plan_detail_for_each_phase/Phase_0_Coursework_contract.md
COURSE_WORK/docs/plan-doc/plan_detail_for_each_phase/Phase_1_Environment.md
COURSE_WORK/docs/plan-doc/plan_detail_for_each_phase/Phase_2_Data_acquisition.md
COURSE_WORK/docs/plan-doc/plan_detail_for_each_phase/Phase_3_Schema_audit.md
COURSE_WORK/docs/plan-doc/plan_detail_for_each_phase/Phase_4_Temporal_integrity_audit.md
COURSE_WORK/docs/plan-doc/plan_detail_for_each_phase/Phase_5_EDA.md
COURSE_WORK/notebook_course_work/CourseWork.ipynb
COURSE_WORK/data/raw_data/energydata_complete.csv
COURSE_WORK/requirements.txt
COURSE_WORK/src/course_work/data/*.py
COURSE_WORK/src/course_work/attention/*.py
```

## 4. Baseline evidence

### 4.1. Notebook identity

```text
Path: COURSE_WORK/notebook_course_work/CourseWork.ipynb
SHA-1: 353e8ea2db2713041ae8aa34903907a40bce93bd
Git state: UNTRACKED
Notebook format: 4.5
Kernel name: python3
Kernel display name: venv (3.10.11.final.0)
Cell count: 43
Code cells: 37
Markdown cells: 6
Unexecuted code cells: 3
Stored error outputs: 0
Stored warning streams: 1
Stored PNG outputs: 11
```

Unexecuted cell indexes:

```text
1
10
26
```

These cells create values used by later cells:

```text
Cell 1 creates df.
Cell 10 creates hour, day_of_week, day_name and is_weekend.
Cell 26 creates corr_matrix.
```

Later cells contain stored outputs that depend on these values. Therefore the stored notebook state does not prove a clean top-to-bottom execution.

### 4.2. Execution-order evidence

The first stored execution counts include:

```text
46, 4, 7, 6, 9, 10, 11, 12
```

Execution counts are not monotonic. The output state is stale or assembled from multiple kernel sessions.

### 4.3. Environment drift evidence

One stored warning contains a Windows temporary path:

```text
C:\Users\minhchi\AppData\Local\Temp\ipykernel_9924\...
```

The current workspace is on macOS. The notebook has no environment report linking the stored output to the current runtime.

### 4.4. Raw data identity

```text
Path: COURSE_WORK/data/raw_data/energydata_complete.csv
Size: 11,979,363 bytes
Rows including header: 19,736
SHA-1: 346ce24141cff53b76bb74a87832c916bd23c860
SHA-256: 2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d
```

The raw CSV header begins:

```text
date, Appliances, lights, T1, RH_1, ...
```

The DataFrame shown by the notebook begins:

```text
date, lights, T1, RH_1, ...
```

and places `Appliances` last because the notebook constructs the DataFrame by concatenating `ucimlrepo` features and targets. This is not the canonical raw column order required by Phase 3.

### 4.5. Project-state evidence

```text
COURSE_WORK/requirements.txt is empty.
No artifacts directory exists.
No Phase 0-5 manifest exists.
No Phase 0-5 sign-off exists.
No checksum file exists.
No acquisition README exists.
No schema artifact exists.
No temporal artifact exists.
No EDA artifact exists.
Current Python source files are empty scaffolds.
```

## 5. Governance blocker

`rule_code.md` declares an architecture rule mandatory before implementation, but neither of these files currently exists:

```text
COURSE_WORK/docs/RULE_BASE/architecture_rule.md
COURSE_WORK/docs/RULE_BASE/rule_process_with_architecture.md
```

The implementation cannot claim architecture conformance until a canonical architecture contract is restored or created and approved.

## 6. Phase 0 finding

Status:

```text
NOT_IMPLEMENTED
```

Missing mandatory content:

```text
Problem statement
RQ1-RQ7
Task and target contract
Forecast horizon contract
Lookback registry
Feature-set registry
Time-feature registry
Target-scaling registry
Boundary protocol registry
Model registry
Metric policy
Validation and Test policy
Fair-comparison rules
Attention rules
Protocol violations
Definition of Done
Machine-readable contract
Phase sign-off
```

The notebook has six Markdown cells in total, but all are generic EDA headings. Phase 0 alone requires six to eight structured Markdown cells.

Impact:

```text
No downstream Phase has an approved scientific contract.
Phase 1 must not be treated as started.
All downstream outputs are non-signoff exploratory work.
```

## 7. Phase 1 finding

Status:

```text
NOT_IMPLEMENTED
```

Missing mandatory content:

```text
Interpreter identity
Kernel identity verification
Version inventory
Platform inventory
CUDA, MPS and CPU detection
Selected-device policy
Tensor smoke test
Autograd smoke test
nn.Module and optimizer-step smoke test
Seed utility
Randomness smoke test
Reproducibility policy
Environment report
Dependency freeze
ENV-v1
Phase sign-off
```

Impact:

```text
Stored outputs cannot be attributed to a reproducible environment.
Phase 2 precondition is absent.
```

## 8. Phase 2 finding

Status:

```text
PARTIAL_NONCONFORMING
```

Observed implementation:

```text
fetch_ucirepo(id=374)
concatenate features and targets in memory
```

The acquisition cell is currently unexecuted.

Missing mandatory outputs:

```text
Canonical official archive
Archive integrity result
Archive SHA-256
CSV checksum file
Dataset manifest
Source metadata
Variable metadata
README_SOURCE.md
Acquisition log
DATA-v1
Phase sign-off
```

Protocol issue:

```text
Downstream cells use the in-memory ucimlrepo DataFrame instead of the canonical local raw CSV.
```

Impact:

```text
Phase 3 is not auditing the canonical raw artifact.
Column-order evidence differs from the raw CSV.
Dataset provenance is incomplete.
```

## 9. Phase 3 finding

Status:

```text
PARTIAL_NONCONFORMING
```

Implemented fragments:

```text
df.info()
df.describe()
missing-value counts
timestamp conversion
```

Missing mandatory checks:

```text
DATA-v1 checksum verification
Expected-versus-actual ordered columns
Duplicate column names
Whitespace and case audit
Semantic role mapping
Feature-group mapping
Unit dictionary
Numeric coercion
Infinite values
All-null columns
Constant columns
Random-control classification
Feature-count reconciliation
Schema fingerprint
Schema manifest
Discrepancy log
SCHEMA-v1
Phase sign-off
```

Protocol violations:

```text
The notebook mutates df['date'] instead of performing a parse probe.
The parsing format omits the required space between date and time.
The audited DataFrame does not preserve canonical raw column order.
```

## 10. Phase 4 finding

Status:

```text
NOT_IMPLEMENTED
```

Missing mandatory checks and outputs:

```text
DATA-v1 and SCHEMA-v1 verification
Deep-copy derived temporal view
Strict timestamp parsing report
Original-order audit before sorting
Negative and zero deltas
Duplicate timestamp classification
Expected 10-minute interval
Delta distribution
Grid alignment
Expected grid
Missing timestamps
Gap table
Coverage metrics
Hourly and daily counts
Continuity segments
Window-safety contract
Temporal manifest
Discrepancy log
TEMPORAL-v1
Phase sign-off
```

Impact:

```text
Phase 5 has no verified temporal input.
Future window construction cannot reject gap-crossing samples.
```

## 11. Phase 5 finding

Status:

```text
PARTIAL_NONCONFORMING
```

Implemented fragments:

```text
Missingness display
Basic target distribution
Daily target average and one-week plot
Sampling counts by hour and weekday
Correlation heatmap
Target-correlation ranking
Selected-feature cross-correlation
Weekday and hour boxplots
IQR-based sensor outlier counts
```

Missing mandatory analyses:

```text
Input-version verification
Dedicated df_eda derived view
Target ECDF
Target boxplot
Full target timeline
Deterministic 24-hour and 7-day windows
Spike analysis
Mean and median energy profiles
Hour-by-weekday energy heatmap
Temperature distributions
Humidity distributions
Weather distributions
lights, rv1 and rv2 summaries
High-correlation feature pairs
Target autocorrelation
Segment-aware lag analysis
Segment-aware rolling mean and standard deviation
Extreme-target samples
Hypothesis registry
Anomaly log
Tables
Named figures
EDA manifest
EDA-v1
Phase sign-off
```

Protocol violations:

```text
Temporal analytical columns are added to df instead of df_eda.
IQR outliers are replaced with NaN.
The resulting values are linearly interpolated.
This is preprocessing, not EDA.
The interpolation may use future observations.
No continuity-segment protection exists.
```

## 12. Architecture finding

The notebook currently owns:

```text
Data acquisition
Schema manipulation
Temporal conversion
EDA transformations
Function definitions
Correlation computation
Outlier transformation
Interpolation
Figure construction
```

The intended architecture requires reusable processing logic to live under `src/course_work` and the notebook to act as an orchestration and presentation layer.

Current source files are empty, so the notebook cannot call verified public APIs.

## 13. Root-cause synthesis

Primary root cause:

```text
The notebook was developed as a generic exploratory notebook instead of being implemented as six gated Phase contracts with source ownership, machine-readable artifacts and sign-offs.
```

Contributing causes:

```text
Architecture contract is missing.
Phase preconditions were not enforced.
The canonical local raw artifact was not used as the downstream source.
Stored notebook state was accepted without a clean-run gate.
Artifacts and sign-offs were not created.
EDA and preprocessing responsibilities were mixed.
Source modules remained empty.
```

## 14. Impacted downstream scope

Until this issue is resolved, these later phases must remain blocked:

```text
Phase 6 feature engineering
Phase 7 feature variants
Phase 8 chronological split
Phase 9 scaling
Phase 10 windows
All training, evaluation and attention phases
```

## 15. Required resolution order

```text
1. Restore or create the canonical architecture rule.
2. Approve a Phase 0-5 refactor plan.
3. Implement and sign off Phase 0.
4. Implement and sign off Phase 1.
5. Implement and sign off Phase 2.
6. Implement and sign off Phase 3.
7. Implement and sign off Phase 4.
8. Implement and sign off Phase 5.
9. Refactor the notebook to orchestration-only.
10. Perform a clean top-to-bottom verification.
```

No later step may begin when the preceding gate fails.

## 16. Issue acceptance

This issue synthesis is accepted as evidence when:

```text
The notebook hash matches the audited baseline.
The raw-data hash matches the audited baseline.
No Phase 0-5 artifact or sign-off exists.
The missing architecture-rule state is confirmed.
The listed notebook cells and execution state are reproducible from the notebook JSON.
```

## 17. Current disposition

```text
ISSUE_CONFIRMED=true
ROOT_CAUSE_CONFIRMED=true
PHASE_0_PASS=true
PHASE_1_PASS=true
PHASE_2_PASS=true
PHASE_3_PASS_WITH_DOCUMENTED_WARNING=true
PHASE_4_PASS=true
PHASE_5_PASS=true
NOTEBOOK_ORCHESTRATION_ONLY=true
NOTEBOOK_CLEAN_EXECUTION_PASS=true
PHASE_6_START_ALLOWED=true
PHASE_6_STARTED=false
FIX_EXECUTION_ALLOWED=true
```

## 18. Resolution evidence

```text
Phase statuses: PASS, PASS, PASS, PASS_WITH_WARNING, PASS, PASS
Notebook code cells: 8
Notebook execution counts: 1, 2, 3, 4, 5, 6, 7, 8
Notebook error outputs: 0
Notebook warning streams: 0
Signed checksum mismatches: 0
Automated tests: 42/42 PASS
EDA tables: 16/16
EDA figures: 16/16
Raw SHA-256: 2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d
Phase 6 artifacts created: false
```
