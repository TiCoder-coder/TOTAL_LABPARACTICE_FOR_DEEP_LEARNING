# Phase 0–6 Handoff Fix Plan — 2026-08-12

## 1. Objective

Sửa và hoàn thiện Phase 0 → Phase 6 để đạt trạng thái:

**READY FOR VIÊN TO START PHASE 7**

Không thay đổi mục tiêu học thuật của Practice 3. Chỉ sửa lỗi, bổ sung verification và evidence còn thiếu.

## 2. Priority 1 — Make the pipeline runnable

### 2.1 Fix Phase 3 notebook syntax

- Sửa lỗi cú pháp ở cell Phase 3.
- Chạy lại cell.
- Xác minh notebook không dừng ở Phase 3.

### 2.2 Normalize imports

- Kiểm tra toàn bộ `processing_own_phase/`.
- Chuẩn hóa import để các module chạy đúng khi notebook import package.
- Không còn `ModuleNotFoundError` do `config`.

### 2.3 Verify dependencies/environment

Xác minh/cài nếu thiếu:

- `transformers`
- `datasets`
- `evaluate`
- `accelerate`
- `torch`
- `numpy`

Phase 1 phải phản ánh environment/device hiện tại.

## 3. Priority 2 — Complete Phase 3–6 requirements

### 3.1 Phase 3 — Tokenization

Notebook phải hiển thị thực tế:

- raw sentence;
- tokens;
- token IDs;
- attention mask;
- special-token information;
- decoded text;
- decode sanity check.

### 3.2 Phase 4 — Dataset Loading

Bổ sung/chạy:

- train/validation/test existence check;
- expected split size check;
- label-set validation trên cả 3 split;
- assert labels subset `{0,1}`.

### 3.3 Phase 5 — EDA & Sanity Checks

Notebook phải chạy:

- schema inspection;
- null/empty/whitespace checks;
- label distribution;
- duplicate analysis;
- character/word/token length;
- P95;
- P99;
- max token length;
- token-length histogram.

Chỉ sau đó mới chốt `MAX_TOKEN_LENGTH`.

### 3.4 Fix RESULT_DIR

Result phải lưu đúng:

```text
total_practice/practice_3/docs/result/
```

### 3.5 Phase 6 — Preprocessing

Bổ sung/chạy:

- tokenization cho train/validation/test;
- sample-count preservation;
- `input_ids` existence;
- `attention_mask` existence;
- label validity;
- sequence-length constraints;
- decode sanity check;
- dynamic-padding verification.

Dynamic-padding test phải dùng batch có sequence dài/ngắn khác nhau và xác minh padding + attention mask thực tế.

Không in PASS vô điều kiện.

## 4. Priority 3 — Evidence, Handoff, Commit

### 4.1 Run Phase 0 → Phase 6

Chạy notebook từ đầu đến hết Phase 6 trong kernel/environment hiện tại.

### 4.2 Save result artifacts

Lưu tối thiểu:

- environment summary;
- dataset summary;
- EDA/token-length statistics;
- token-length histogram;
- preprocessing verification summary.

### 4.3 Update handoff

Cập nhật:

```text
docs/result/phase_06_handoff_2026-08-12.md
```

Chỉ ghi PASS từ evidence vừa chạy.

Phải có:

- branch;
- commit hash;
- dataset counts;
- tokenizer/checkpoint;
- max_length justification;
- padding strategy;
- sanity-check results;
- known issues;
- files modified.

### 4.4 Commit and push

Suggested commit:

```text
fix(practice3): verify and complete phases 0-6 handoff
```

### 4.5 Re-audit

Target cuối:

- Phase 0–6: PASS
- Workflow compliance: PASS
- Phase 6 handoff: READY FOR VIÊN TO START PHASE 7

## 5. Files Expected to Change

Potential files:

- `notebook_practice_3/practice_3.ipynb`
- `processing_own_phase/config.py`
- `processing_own_phase/phase_01_environment.py`
- `processing_own_phase/phase_03_tokenization.py`
- `processing_own_phase/phase_04_dataset_loading.py`
- `processing_own_phase/phase_05_dataset_eda.py`
- `processing_own_phase/phase_06_preprocessing.py`
- `docs/result/*`
- `docs/result/phase_06_handoff_2026-08-12.md`

Không chỉnh Phase 7 trở đi trong vòng fix này.

## 6. Stop Condition

Dừng sau khi:

1. Phase 0–6 chạy thành công.
2. Evidence được lưu.
3. Handoff được cập nhật.
4. Audit lại hoàn tất.

**Không bắt đầu Phase 7 cho đến khi audit xác nhận READY.**
