# COURSEWORK PHASE 5 DIRECT NOTEBOOK EDA INTEGRATION PLAN

## Plan metadata

```text
Plan ID: CW-PHASE-5-EDA-DIRECT-001
Supersedes: CW-PHASE-5-EDA-COVERAGE-001
Execution mode: STRICTLY_SEQUENTIAL
Status: COMPLETED
```

## Objective

Tích hợp trực tiếp các bước EDA không trùng từ `EDA.ipynb` vào phần Phase 5 của `CourseWork.ipynb` theo override mới của Human, đồng thời giữ nguyên thứ tự Phase 0 đến Phase 5 và không thay đổi dữ liệu hoặc artifact canonical.

## Human override

```text
Cho phép pandas, NumPy, matplotlib và seaborn EDA code trong Phase 5 notebook cells.
Cho phép DataFrame derivation, descriptive calculation và plotting trực tiếp trong notebook.
Không yêu cầu tách EDA bổ sung sang file Python riêng.
```

Override chỉ áp dụng cho exploratory presentation trong Phase 5. Nó không cho phép notebook sở hữu training, preprocessing canonical, split, scaling, windowing hoặc modeling logic.

## Scope

```text
Data overview và df.info.
Descriptive statistics.
Missing-value counts, percentages và heatmap.
EDA-only calendar columns.
Sampling counts theo hour, weekday và weekend.
Target distribution và temporal views.
Correlation ranking, heatmap và top-feature scatterplots.
Cross-correlation profile ±24 steps.
Conditional target boxplots.
IQR outlier diagnostics.
Isolated outlier-smoothing demonstration trên deep copy.
```

## Duplicate policy

Các figure và table đã có trong `EDA-v1` tiếp tục được hiển thị từ canonical artifacts. Không tính lại các output đó nếu direct notebook step không bổ sung thông tin mới.

## Safety boundary

```text
Không fetch lại UCI dataset.
Không ghi file từ direct EDA cells.
Không overwrite raw CSV hoặc artifacts.
Không dùng cleaned demonstration DataFrame cho Phase 6 hoặc model input.
Không thay đổi `df` canonical khi minh họa smoothing.
Không thêm code comments hoặc icons.
Không thêm machine-specific path.
Không thay đổi Phase 0-5 materialization calls.
```

## Impacted files

```text
COURSE_WORK/notebook_course_work/CourseWork.ipynb
COURSE_WORK/docs/RULE_BASE/architecture_rule.md
COURSE_WORK/tests/integration/test_notebook_boundary.py
COURSE_WORK/docs/plan-doc/plan_detail_for_each_phase/Phase_5_EDA.md
```

## Sequential execution

### Step 1

Record raw CSV, Phase 0-5 sign-off and notebook baseline checksums.

### Step 2

Update architecture documentation so the direct EDA exception is narrow, explicit and cannot spread into later phases.

### Step 3

Update notebook-boundary tests to allow direct descriptive EDA only inside the Phase 5 section while continuing to forbid network acquisition, file writes, path injection, canonical preprocessing and model logic.

### Step 4

Refactor Phase 5 notebook cells in ordered subsections. Reuse canonical Phase 5 artifacts for duplicate outputs and implement missing exploratory views directly.

### Step 5

Run notebook structure and boundary tests. Stop and repair before proceeding if any test fails.

### Step 6

Run the full automated suite and verify canonical artifact checksums and raw CSV fingerprint.

### Step 7

Clear notebook state, execute top to bottom in a clean project kernel and validate every output.

## Validation

```text
Notebook JSON and Python syntax valid.
Phase headings remain ordered.
Every reference EDA group is present or explicitly covered by an existing artifact.
No fetch_ucirepo call exists.
No direct EDA cell writes files.
No comment or icon exists in code cells.
Canonical raw DataFrame is not mutated by smoothing demonstration.
No notebook error or warning output exists.
All automated tests pass.
Raw CSV hash is unchanged.
Phase 0-5 sign-off hashes are unchanged.
Phase 6 remains unstarted.
```

## Completion result

```text
CourseWork.ipynb contains 14 ordered Phase 5 EDA subsections.
All 21 notebook code cells execute successfully in sequence after the restored notebook was rebuilt.
Data overview output includes df.info for all 29 canonical columns.
The notebook embeds 16 canonical and 8 direct EDA PNG figures.
No notebook error or warning stream remains.
The full automated suite passes with 47 tests.
Raw data and Phase 0-5 sign-off checksums remain unchanged.
```
