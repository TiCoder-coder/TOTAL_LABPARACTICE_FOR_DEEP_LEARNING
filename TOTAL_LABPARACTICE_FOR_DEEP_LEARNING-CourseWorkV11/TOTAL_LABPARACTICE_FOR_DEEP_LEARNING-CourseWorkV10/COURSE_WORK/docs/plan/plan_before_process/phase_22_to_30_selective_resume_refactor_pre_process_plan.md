# Phase 22 to Phase 30 Selective Resume Refactor Pre-Process Plan

## 1. Plan identity

```text
Plan ID: CW-PHASE-22-30-SELECTIVE-RESUME-PREPROCESS-v1
Repository scope: COURSE_WORK
Phase scope: Phase 22 through Phase 30
Plan type: Architecture, notebook orchestration, selective resume and persistent presentation
Approval state: APPROVED_AND_EXECUTED
Scientific execution state: NOT_AUTHORIZED
```

Tài liệu này ghi lại kế hoạch đã được chốt trước khi refactor workflow Phase 22 đến Phase 30. Kết quả thực thi tương ứng được lưu tại:

```text
docs/plan-doc/plan_to_refactor&fix/phase_22_to_30_selective_resume_refactor_execution_report.md
```

## 2. Bối cảnh trước khi xử lý

- Phase 22 đến Phase 29 đọc CSV và xử lý hiển thị trực tiếp trong notebook.
- Phase 23 đến Phase 29 chứa conditional flow và Matplotlib logic trong notebook.
- Một số nhánh thiếu file tham chiếu dataframe chưa được khởi tạo.
- Phase 30 phụ thuộc trạng thái runtime của các cell trước.
- Signoff PASS chưa bắt buộc xác minh đầy đủ declared artifact và checksum.
- Pending conditions được xác định bằng danh sách cứng.
- Entry point chạy condition gọi lại materializer Phase 0 đến Phase 13.
- Output cuối notebook sử dụng widget MIME không thể khôi phục sau khi mở lại.
- Chưa có contract bảo toàn output của các cell ngoài phạm vi refactor.

## 3. Mục tiêu

Refactor workflow để một Phase muộn có thể được kiểm tra độc lập sau khi restart kernel mà không chạy lại toàn bộ notebook.

Hệ thống phải:

1. phân biệt canonical scientific artifact với derived processing log;
2. kiểm tra dependency và evidence theo chế độ read-only;
3. chỉ tái sử dụng signoff khi file, checksum, condition và lineage hợp lệ;
4. xác định công việc còn thiếu bằng expected conditions trừ verified conditions;
5. không gọi materializer upstream khi chỉ yêu cầu Phase 23 đến Phase 30;
6. chỉ cho phép chạy condition thiếu khi environment và prerequisite hợp lệ;
7. hiển thị kết quả bằng HTML tĩnh có thể lưu trong notebook;
8. giữ nguyên output của mọi cell ngoài phạm vi được duyệt;
9. không chạy training trong phạm vi plan này.

## 4. Phạm vi Phase

| Phase | Chức năng | Notebook sau refactor | Selective condition contract |
|---|---|---|---|
| 22 | Learning-Curve Diagnostics | `render_phase_summary(22, PROJECT_ROOT)` | Không áp dụng |
| 23 | S1 Feature-Set Sweep | `render_phase_resume(23)` | `FS0_TF1`, `FS1_TF1`, `FS2_TF1` |
| 24 | S2 Time-Feature Sweep | `render_phase_resume(24)` | `TF0`, `TF1` |
| 25 | S3 Target-Scaling Sweep | `render_phase_resume(25)` | `YS0`, `YS1` |
| 26 | S4 Lookback Sweep | `render_phase_resume(26)` | `L36`, `L72`, `L144` |
| 27 | S5 Pooling Sweep | `render_phase_resume(27)` | `LAST_STEP`, `MEAN` |
| 28 | S6 Activation Sweep | `render_phase_resume(28)` | `RELU`, `GELU` |
| 29 | S7 Batch-Size Sweep | `render_phase_resume(29)` | `B32`, `B64` |
| 30 | S8 Learning-Rate Sweep | `render_phase_resume(30)` | `LR1=0.0001`, `LR2=0.0003`, `LR3=0.001` |

## 5. Architecture ownership

### 5.1. Notebook

`notebook_course_work/CourseWork.ipynb` chỉ sở hữu thứ tự trình bày, public API calls, narrative và stored HTML output.

Notebook không sở hữu CSV loading, checksum validation, condition resolution, dependency validation, training hoặc plotting logic của Phase 22 đến Phase 30.

### 5.2. Selective execution

`src/course_work/experiments/phase_execution.py` sở hữu:

- Phase 23 đến Phase 30 registry;
- prerequisite registry;
- expected condition registry;
- canonical evidence inspection;
- environment readiness inspection;
- state classification;
- action resolution;
- expected-minus-verified resolution;
- condition-level execution gate.

### 5.3. Signoff validation

`src/course_work/sweeps/sweep_results.py` phải từ chối signoff khi declared file bị thiếu, checksum bị thiếu hoặc không khớp, status không hợp lệ, condition coverage thiếu hoặc lineage mâu thuẫn.

### 5.4. Presentation

`src/course_work/reporting/phase_summary.py` sở hữu derived processing log, compact decision tables, prerequisite table, condition evidence table và static aggregate-log HTML.

### 5.5. Execution entry points

```text
scripts/run_single_condition.py
scripts/run_all_pending.py
scripts/run_phase_background.py
scripts/run_phase_background/_notebook.py
scripts/sweep_results_to_csv.py
```

Các entry point phải nhận phase và condition rõ ràng, kiểm tra gate trước khi launch và không materialize upstream một cách ẩn.

## 6. State và action contract

| State | Action |
|---|---|
| `VALID_REUSABLE` | `RENDER_ONLY` |
| `LOG_MISSING` | `REBUILD_LOG_ONLY` |
| `LOG_STALE` | `REBUILD_LOG_ONLY` |
| `DERIVED_ARTIFACT_MISSING` | `REBUILD_DERIVED_ONLY` |
| `CONDITION_INCOMPLETE` | `EXECUTE_MISSING_ONLY` |
| `RUNNING` | `WAIT_FOR_RUNNING_PROCESS` |
| `FAILED` | `EXECUTE_MISSING_ONLY` sau khi gate hợp lệ |
| `SIGNOFF_INVALID` | `BLOCK` |
| `UPSTREAM_INVALID` | `BLOCK` |
| `ENVIRONMENT_INVALID` | `BLOCK` |

## 7. Evidence priority

1. Phase detail và architecture rule đã duyệt.
2. Approved configuration.
3. Experiment registry run configuration.
4. Run artifact và checksum.
5. Phase manifest và signoff.
6. Derived processing log.
7. Stored notebook output.

Log hoặc notebook output không được ghi đè mâu thuẫn ở tầng evidence cao hơn.

## 8. Notebook output preservation plan

Trước khi sửa phải lưu notebook checksum, cell count, cell ID, execution count, output count, MIME type, content hash từng cell và vị trí widget lỗi.

Trong khi sửa:

- không clear output toàn notebook;
- không thay output ngoài cell Phase 30 và final aggregate cell;
- Phase 22 đến Phase 29 chỉ thay source và giữ stored output;
- Phase 30 được thay output bằng recovery-gate HTML;
- final aggregate cell được thay widget bằng static HTML.

## 9. HTML presentation contract

HTML phải:

- tĩnh và persistent;
- không có widget MIME;
- không có JavaScript runtime;
- không yêu cầu external asset;
- escape dữ liệu;
- dùng compact table và responsive overflow;
- chỉ hiển thị thông tin phục vụ quyết định;
- giữ chi tiết kỹ thuật trong JSON log.

Phase resume view phải hiển thị state, action, execution authorization, prerequisite validation, condition evidence và block reason.

## 10. Trình tự xử lý và verification

### Step 1. Capture preservation baseline

Verify notebook parse, cell ID uniqueness, widget location và reproducible hashes.

### Step 2. Cập nhật architecture contract

Verify Phase ownership, evidence hierarchy và notebook boundary.

### Step 3. Xây dựng read-only inspector

Verify inspector không gọi materializer, phát hiện missing file, checksum mismatch và invalid upstream.

### Step 4. Siết signoff reuse

Verify stale PASS bị từ chối, valid fixture vẫn reusable và Test không bị truy cập.

### Step 5. Xây dựng condition resolver

Verify expected conditions đúng từng Phase, verified condition không chạy lại và chỉ missing conditions được đề xuất.

### Step 6. Refactor execution entry points

Verify phase và condition explicit, dry-run không launch upstream và invalid gate chặn trước process launch.

### Step 7. Refactor static reporting

Verify không có widget MIME, script hoặc raw JSON dump trong notebook output.

### Step 8. Refactor notebook Phase 22 đến Phase 30

Verify mỗi cell chỉ gọi public API, không còn data processing, plotting, control flow hoặc comment.

### Step 9. Chạy Phase 30 recovery gate

Verify chỉ inspector và renderer được gọi, không training và không materialize upstream.

### Step 10. Chạy regression

Verify focused tests, notebook validation, Python compilation, output preservation và patch whitespace.

## 11. Expected immediate result

```text
Phase 30 state: UPSTREAM_INVALID
Phase 30 action: BLOCK
Training authorized: false
```

Nguyên nhân dự kiến là Phase 29 chưa có đầy đủ winner, reference-update và condition evidence hợp lệ.

## 12. Files được phép thay đổi

```text
docs/RULE_BASE/architecture_rule.md
docs/plan-doc/analysis_error/notebook_widget_and_phase_selective_resume_issue.md
docs/plan-doc/plan_before_process/refactor_notebook_selective_phase_resume_plan.md
docs/save_log_in_processing/phase_30_s8_learning_rate_log.json
docs/save_log_in_processing/notebook_output_preservation_baseline.json
docs/save_log_in_processing/notebook_output_content_preservation_baseline.json
notebook_course_work/CourseWork.ipynb
src/course_work/experiments/phase_execution.py
src/course_work/reporting/phase_summary.py
src/course_work/sweeps/sweep_results.py
scripts/run_single_condition.py
scripts/run_all_pending.py
scripts/run_phase_background.py
scripts/run_phase_background/_notebook.py
scripts/sweep_results_to_csv.py
tests/contracts/test_selective_execution_policy.py
tests/unit/test_phase_execution.py
tests/unit/test_selective_phase_reporting.py
tests/unit/test_phase_summary.py
```

## 13. Không được thay đổi

- raw data;
- Test targets hoặc Test evaluation gate;
- Phase ordering;
- model definition;
- shared metric contract;
- signed scientific artifact bằng dữ liệu giả;
- output của notebook cell ngoài phạm vi đã duyệt;
- condition value trong approved Phase detail;
- Phase 29 hoặc Phase 30 scientific result khi chưa có approval riêng.

## 14. Acceptance criteria

- Phase 22 đến Phase 30 notebook cells chỉ còn public calls;
- Phase 30 chạy độc lập sau restart để inspect và render;
- không rerun toàn notebook;
- không chạy lại verified condition;
- missing condition được tính động;
- invalid upstream chặn execution;
- stale PASS bị từ chối;
- final widget được thay bằng static HTML;
- non-target output hash không đổi;
- Test vẫn locked;
- scientific execution không được kích hoạt.

## 15. Approval record

```text
Workflow refactor approved by Human: true
Notebook source refactor approved by Human: true
Notebook target-output replacement approved by Human: true
Architecture amendment approved by Human: true
Scientific recovery approved by Human: false
```
