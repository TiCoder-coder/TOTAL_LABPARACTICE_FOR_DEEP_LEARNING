# Phase 2 Sign-off Checksum Corrective Plan

## 1. Trạng thái plan

```
Plan ID: PHASE-2-SIGNOFF-CHECKSUM-CORRECTIVE-v1
Issue ID: PHASE-2-CHECKSUM-MISMATCH-BLOCKER-v1
Status: READY_FOR_APPROVAL
Execution mode: STRICTLY_SEQUENTIAL
```

## 2. Mục tiêu

- Khôi phục output_checksums trong Phase 2 sign-off khớp với actual artifacts
- Cập nhật input_checksums reference đến Phase 1 sign-off đã khôi phục
- Mở lại Phase 2 verification và downstream chain

## 3. Ownership

- Owner: `src/course_work/data/acquisition.py` và `artifacts/acquisition/`
- Corrective target: `artifacts/acquisition/phase_2_signoff.json`
- Consumer verification: Phase 3 loader, EDA tests

## 4. Bằng chứng nguồn

### 4.1 Output checksum source

acquisition_log.json actual checksum:
```
010695e4487fbd39ea466cdd73bafc2bbd0c6b964f2f688d361076154c5186e7
```

Commit 92ab9df1 sign-off cũng khai báo checksum này.

### 4.2 Input checksum source

Phase 1 sign-off đã khôi phục:
```
db146d0a576aacefcb816667a84abef8e11f727c3ff165cdb20d50d2028fbfbd
```

## 5. Trình tự thực thi

### Step 1. Preservation

1. Ghi checksum canonical sign-off hiện tại
2. Ghi checksum acquisition_log.json hiện tại
3. Verify Phase 1 sign-off đã khôi phục đúng

### Step 2. Update input_checksums

1. Đọc current phase_2_signoff.json
2. Cập nhật `input_checksums.artifacts/environment/phase_1_signoff.json` từ:
   - `f909c62d3b280c7d8f3c14479fa0366bb87cc97d61c9843e069c76085282546f`
   - thành `db146d0a576aacefcb816667a84abef8e11f727c3ff165cdb20d50d2028fbfbd`

### Step 3. Update output_checksums

1. Cập nhật `output_checksums.artifacts/acquisition/acquisition_log.json` từ:
   - `f0202f155d2e253ef27c1673cb49d6bb372c87a74fc95832691b8f87626126ad`
   - thành `010695e4487fbd39ea466cdd73bafc2bbd0c6b964f2f688d361076154c5186e7`

### Step 4. Immediate verification

1. Gọi `materialize_phase_2`
2. Verify không raise RuntimeError
3. Chạy `test_acquisition.py` nếu tồn tại

### Step 5. Downstream verification

1. Chạy `test_eda.py`
2. Verify Phase 1 fail đã biến mất
3. Verify Phase 2 checksum PASS

## 6. File được phép thay đổi

```
artifacts/acquisition/phase_2_signoff.json
```

## 7. File không được thay đổi

```
artifacts/acquisition/acquisition_log.json
artifacts/acquisition/_history/**
src/course_work/data/acquisition.py
src/course_work/data/schema.py
notebook_course_work/CourseWork_1.ipynb
```

## 8. Acceptance criteria

- Phase 2 loader không raise RuntimeError
- EDA tests không bị chặn bởi Phase 2
- Không output nào bị regenerate
- Không có thay đổi source code

## 9. Approval gate

```
ISSUE_CONFIRMED=true
ROOT_CAUSE_IDENTIFIED=true
CORRECTIVE_PLAN_CREATED=true
WAITING_FOR_HUMAN_APPROVAL=true
```
