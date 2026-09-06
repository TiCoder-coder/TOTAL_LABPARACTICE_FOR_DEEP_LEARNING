# Kế hoạch xử lý historical state và full test suite sau Phase 33

## 1. Danh tính kế hoạch

```text
Plan ID: FULL-TEST-HISTORICAL-STATE-RECONCILIATION-v1
Issue source: full_test_suite_historical_state_nonconformance_after_phase_33_visualization_issue.md
Execution status: COMPLETED
Execution mode: Strictly sequential
```

## 2. Mục tiêu

Khôi phục full test suite về trạng thái phản ánh đúng canonical lifecycle Phase 0-33 mà không sửa tay artifact, không làm yếu scientific contract và không phá Test firewall.

## 3. Non-goals

```text
Không thay model hoặc hyperparameter.
Không chạy lại sweep.
Không mở Test.
Không thay current reference.
Không sửa task visualization Phase 33 đã pass.
Không tự động regenerate artifact trước khi xác định producer ownership.
```

## 4. Trình tự đề xuất

### Step 1. Preservation baseline toàn repository state liên quan

Ghi nhận checksum, git state và lineage của:

```text
Phase 2 acquisition outputs
Phase 13 registry outputs
Phase 15-22 artifacts
Phase 15-22 processing logs
Environment report và Phase 1 sign-off
```

### Step 2. Phân loại canonical revision và stale artifact

Xác định cho từng mismatch:

```text
file hiện tại là canonical mới hay thay đổi ngoài contract
sign-off nào sở hữu checksum
producer nào được phép regenerate
downstream artifact nào phụ thuộc checksum đó
```

Không sửa file ở bước này.

Kết quả audit ngày 2026-08-23:

```text
Phase 2 sign-off, acquisition log và Phase 3 lineage cùng xác nhận producer revision 2026-08-20.
data/raw_data/README_SOURCE.md hiện là metadata revision 2026-08-14.
data/raw_data/dataset_manifest.json hiện là metadata revision 2026-08-14.
Raw CSV checksum vẫn đúng approved baseline.
ZIP checksum vẫn đúng approved baseline.
Canonical README bytes tái tạo từ acquisition_log đạt signed checksum.
Canonical manifest bytes tái tạo từ acquisition_log đạt signed checksum.
Phase 15-22 canonical artifacts hợp lệ nhưng processing logs đang stale.
Registry hiện có 19 canonical run records; fixed expectation 1 đã lỗi thời.
Read-only data tests đang gọi materialize_phase_1 và bị chặn bởi training-only GPU gate.
```

### Step 3. Revision test expectation của experiment registry

Thay assertion số run cố định bằng contract kiểm tra:

```text
registry run_count khớp số record canonical
Persistence run tồn tại đúng một canonical identity
mọi run ID duy nhất
mọi lifecycle state hợp lệ
Phase 13 baseline contract vẫn được bảo toàn
```

Không được giảm coverage hoặc chỉ xóa assertion đang fail.

### Step 4. Thiết kế test-only environment boundary

Phân tách rõ:

```text
training execution phải có GPU theo policy hiện tại
read-only artifact validation không được buộc chạy lại training smoke path
unit test pure data logic không được phụ thuộc GPU nếu không kiểm tra GPU behavior
GPU enforcement test phải được kiểm tra riêng bằng mock hoặc explicit environment contract
```

Mọi thay đổi environment policy cần approval riêng nếu làm thay runtime behavior.

### Step 5. Regenerate artifact đúng producer

Chỉ sau khi Step 2 xác nhận canonical source:

```text
archive current Phase 2 metadata revision
recover README_SOURCE.md và dataset_manifest.json bằng acquisition producer từ signed acquisition log
không thay raw CSV hoặc ZIP
verify recovered bytes khớp Phase 2 signed output checksums
regenerate processing log Phase 15-22 bằng reporting producer nếu source artifacts là canonical
không sửa checksum field trực tiếp
verify reload và checksum sau từng Phase
```

### Step 6. Regression tuần tự

```text
Phase 2 integrity tests
Phase 13 registry tests
Phase 15-22 presentation integrity tests
Environment boundary tests
EDA and splitting unit tests
Full test suite
Artifact checksum audit
Notebook output preservation audit
```

## 5. Files dự kiến có thể thay đổi

Danh sách chính xác chỉ được khóa sau Step 2. Candidate scope:

```text
tests/integration/test_phase_0_to_5_chain.py
tests/integration/test_phase_0_to_8_presentation.py
tests/unit/test_eda.py
tests/unit/test_splitting.py
tests liên quan environment policy
src/course_work/data/acquisition.py
src/course_work/utils/environment.py
data/raw_data/README_SOURCE.md
data/raw_data/dataset_manifest.json
artifacts/acquisition/_history/**
producer source của artifact được xác nhận stale
processing logs được regenerate qua canonical reporting API
```

Raw metadata chỉ được thay qua verified producer recovery. Raw CSV và ZIP không được thay đổi.

## 6. Files không được thay đổi

```text
Phase 33 winner/reference/sign-off
Current reference run config
Model checkpoints
Validation metrics
Test artifacts
CourseWork.ipynb visualization đã pass
```

## 7. Acceptance criteria

```text
Không còn fixed run_count expectation mâu thuẫn lifecycle Phase 33.
Mọi signed checksum khớp canonical output.
Mọi processing log source checksum khớp source artifact.
GPU policy được kiểm tra đúng boundary và không bị vô hiệu hóa.
Full test suite pass trong environment contract đã xác định.
Không artifact nào được sửa tay.
Không comment hoặc icon mới trong code.
```

## 8. Stop conditions

```text
Không xác định được canonical side của checksum mismatch.
Producer regeneration làm thay scientific result.
Environment correction yêu cầu thay runtime training policy.
Artifact history không đủ để xác định revision hợp lệ.
Phát hiện Test access hoặc downstream lineage bị ảnh hưởng.
```

## 9. User approval gate

```text
PLAN_CREATED=true
IMPLEMENTATION_STARTED=true
WAITING_FOR_USER_APPROVAL=false
USER_APPROVAL_GRANTED=2026-08-23
EXECUTION_COMPLETED=2026-08-23
FULL_TEST_RESULT=274_PASSED_4_SUBTESTS_PASSED
```
