# Phase 1 Sign-off Checksum Provenance Corrective Plan

## 1. Trạng thái plan

```text
Plan ID: PHASE-1-SIGNOFF-CHECKSUM-PROVENANCE-CORRECTIVE-v1
Issue ID: PHASE-1-SIGNOFF-CHECKSUM-PROVENANCE-CORRUPTION-v1
Status: WAITING_FOR_HUMAN_APPROVAL
Execution mode: STRICTLY_SEQUENTIAL
```

Plan này chỉ cho phép exact-byte recovery của canonical Phase 1 sign-off. Chưa được thực thi corrective trước khi Human duyệt.

## 2. Mục tiêu

- Khôi phục quan hệ checksum giữa canonical Phase 1 sign-off và ba output đã tồn tại.
- Giữ nguyên scientific environment payload hiện tại.
- Không thay runtime policy, strict validation hoặc downstream artifacts.
- Mở lại strict Phase 1 validation để Stage 0 có thể hoàn tất.

## 3. Ownership

- Owner: `src/course_work/utils/environment.py` và `artifacts/environment/`.
- Corrective target duy nhất: `artifacts/environment/phase_1_signoff.json`.
- Consumer verification: Phase 1 loader, Phase 1 tests và EDA test gọi upstream chain.
- Notebook không tham gia corrective này.

## 4. Bằng chứng nguồn

Nguồn exact recovery được chọn:

```text
Git revision: 1f648662980c49e0ee42de9f2cf43639e6966b64
Path: COURSE_WORK/artifacts/environment/phase_1_signoff.json
```

Revision này có cùng `environment_revision_id`, `created_at`, input lineage và output paths với canonical sign-off hiện tại. Khác biệt nội dung có ý nghĩa chỉ là ba output checksum đã bị commit sau thay sai.

## 5. Trình tự thực thi

### Step 1. Preservation

1. Ghi checksum canonical sign-off hiện tại.
2. Ghi checksum ba current Phase 1 outputs.
3. Xác minh current outputs khớp source revision đã chọn.
4. Xác minh worktree không có Human edit chưa nhận diện tại corrective target.

### Step 2. Candidate construction

1. Đọc exact sign-off blob từ source revision.
2. Kiểm tra JSON schema và Phase identity.
3. Kiểm tra `environment_revision_id` khớp ba current outputs.
4. Kiểm tra từng declared checksum trên candidate khớp current canonical output.
5. Không sửa candidate bằng cách tính rồi chèn checksum mới.

### Step 3. Exact canonical recovery

1. Chỉ thay canonical `phase_1_signoff.json` bằng nội dung exact source blob đã xác minh.
2. Không sửa ba Phase 1 output files.
3. Không sửa `_history`.
4. Không gọi Phase 1 materializer hoặc recovery producer trong bước ghi.

### Step 4. Immediate local verification

1. Xác minh recovered file checksum khớp exact source blob.
2. Gọi `load_validated_environment_signoff`.
3. Xác minh loader trả Phase 1 `PASS`.
4. Chạy `tests/unit/test_environment.py`.
5. Chạy `tests/unit/test_environment_recovery.py`.

### Step 5. Linked upstream and downstream verification

1. Chạy lại `tests/unit/test_eda.py`.
2. Xác minh failure Phase 1 đã biến mất.
3. Xác minh checksum toàn bộ Phase 9, Phase 47, Phase 58, Phase 59 và notebook không đổi so với preservation baseline.
4. Xác minh không có code, comment, docstring hoặc icon mới.

### Step 6. Stage 0 decision

- Nếu tất cả verification PASS, đóng corrective và quay lại Stage 0 của master plan.
- Nếu candidate không khớp exact blob hoặc loader vẫn fail, dừng và tạo issue mới.
- Nếu EDA còn fail vì root cause khác, dừng trước Stage 1 và lập corrective plan riêng.

## 6. File được phép thay đổi

```text
artifacts/environment/phase_1_signoff.json
docs/plan-doc/plan_to_refactor&fix/phase_1_signoff_checksum_provenance_corrective_result.md
```

## 7. File không được thay đổi

```text
artifacts/environment/environment_report.json
artifacts/environment/requirements_freeze.txt
artifacts/environment/smoke_test_report.json
artifacts/environment/_history/**
src/course_work/utils/environment.py
src/course_work/data/eda.py
notebook_course_work/CourseWork_1.ipynb
artifacts/scaling/**
artifacts/final_test/**
artifacts/final_tables/**
artifacts/final_conclusions/**
```

## 8. Acceptance criteria

- Exact source blob được phục hồi, không phải checksum được viết lại tùy ý.
- Phase 1 strict loader PASS.
- Phase 1 unit và recovery tests PASS.
- EDA test không còn bị chặn bởi Phase 1 sign-off.
- Không output Phase 1 nào bị regenerate.
- Không frozen downstream artifact hoặc notebook nào thay đổi.
- Không có thay đổi source code.

## 9. Approval gate

```text
ISSUE_CONFIRMED=true
CORRECTIVE_PLAN_CREATED=true
IMPLEMENTATION_STARTED=false
WAITING_FOR_HUMAN_APPROVAL=true
```
