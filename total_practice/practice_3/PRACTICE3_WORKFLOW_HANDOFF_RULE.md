# PRACTICE 3 --- WORKFLOW & HANDOFF RULE

## 1. Mục đích

Tài liệu này quy định quy trình làm việc, phạm vi nhiệm vụ và cách bàn
giao giữa **Duyên** và **Viên** trong Practice 3 --- Hugging Face
Transformers.

Flow tổng quát:

**Duyên (Phase 0--6) → Verify → Commit/Push → Handoff → Viên Verify →
Viên (Phase 7--14) → Cả hai (Phase 15 + Final Review)**

------------------------------------------------------------------------

## 2. Phân chia nhiệm vụ

### Duyên --- Stage 1

Duyên phụ trách:

-   Phase 0 --- Practice Overview
-   Phase 1 --- Environment & Reproducibility
-   Phase 2 --- Pretrained Sentiment Inference
-   Phase 3 --- Tokenization Investigation
-   Phase 4 --- Dataset Loading
-   Phase 5 --- Dataset EDA & Sanity Checks
-   Phase 6 --- Tokenizer & Preprocessing

### Viên --- Stage 2

Viên phụ trách:

-   Phase 7 --- Model Construction
-   Phase 8 --- Metrics & Training Configuration
-   Phase 9 --- Fine-Tuning
-   Phase 10 --- Learning Curves
-   Phase 11 --- Validation & Test Evaluation
-   Phase 12 --- Error Analysis
-   Phase 13 --- New-Sentence Inference
-   Phase 14 --- Save & Reload Model

### Cả hai

Cùng phụ trách:

-   Phase 15 --- Final Summary
-   Final Review
-   Run All notebook
-   Hoàn thiện `current_flow`
-   Final Audit

------------------------------------------------------------------------

## 3. Quy trình xử lý một Phase

Mỗi Phase phải đi theo flow:

``` text
PLAN
  ↓
IMPLEMENT
  ↓
NOTEBOOK CALL
  ↓
VERIFY OUTPUT
  ↓
SAVE RESULT
  ↓
COMMIT
```

Nếu xảy ra lỗi:

``` text
ERROR
  ↓
ANALYZE ERROR
  ↓
SAVE ERROR DOCUMENT
  ↓
CREATE FIX / REFACTOR PLAN
  ↓
FIX
  ↓
RE-RUN
  ↓
VERIFY
```

------------------------------------------------------------------------

## 4. Plan Before Process

Trước khi thực hiện một Phase hoặc thay đổi quan trọng, người phụ trách
phải tạo plan tại:

``` text
docs/plan-doc/plan_before_process/
```

Quy tắc đặt tên:

``` text
phase_XX_<phase_name>_plan_YYYY-MM-DD.md
```

Ví dụ:

``` text
phase_05_dataset_eda_plan_2026-08-10.md
```

Plan nên ghi tối thiểu:

-   Objective
-   Input
-   Processing
-   Expected Output
-   Files affected
-   Validation / Sanity Checks
-   Completion Criteria

------------------------------------------------------------------------

## 5. Quy tắc implementation

Logic xử lý chính của từng Phase đặt tại:

``` text
processing_own_phase/
```

Ví dụ:

``` text
phase_01_environment.py
phase_02_pretrained_inference.py
phase_03_tokenization.py
```

File notebook chính:

``` text
notebook_practice_3/practice_3.ipynb
```

Không đưa toàn bộ processing logic trực tiếp vào notebook.

Notebook chủ yếu dùng để:

-   import/call module;
-   hiển thị output;
-   hiển thị bảng;
-   hiển thị biểu đồ;
-   giải thích và nhận xét kết quả.

------------------------------------------------------------------------

## 6. Error Documentation

Khi gặp lỗi cần phân tích, tạo file tại:

``` text
docs/plan-doc/analysis_error/
```

Naming:

``` text
phase_XX_<error_name>_YYYY-MM-DD.md
```

Nội dung nên có:

-   Error
-   Phase
-   Context
-   Root cause / suspected cause
-   Evidence
-   Proposed solution
-   Final status

------------------------------------------------------------------------

## 7. Refactor / Fix

Nếu implementation đã được thực hiện nhưng cần thay đổi hoặc refactor,
tạo plan tại:

``` text
docs/plan-doc/plan_to_refactor&fix/
```

Không thực hiện refactor lớn mà không có plan.

Tên file `.md` phải có ngày tháng năm.

------------------------------------------------------------------------

## 8. AI Process Log

Prompt và log quan trọng khi sử dụng AI lưu tại:

``` text
docs/save_process_proceduce_own_phase_refactor&fix/
```

Ưu tiên lưu:

-   implementation prompt;
-   debugging prompt;
-   refactor prompt;
-   audit prompt;
-   các quyết định quan trọng.

------------------------------------------------------------------------

## 9. Result

Kết quả quan trọng từ notebook được lưu hoặc liên kết tại:

``` text
docs/result/
```

Ví dụ:

-   dataset statistics;
-   token-length statistics;
-   figures;
-   training metrics;
-   evaluation metrics;
-   confusion matrix;
-   error-analysis results.

Không fabricate kết quả. Kết quả phải đến từ code/notebook đã thực thi.

------------------------------------------------------------------------

## 10. Stage 1 --- Quy trình của Duyên

Duyên thực hiện tuần tự:

``` text
Phase 0
   ↓
Phase 1
   ↓
Phase 2
   ↓
Phase 3
   ↓
Phase 4
   ↓
Phase 5
   ↓
Phase 6
```

Không bàn giao cho Viên khi Phase 6 chưa hoàn thành và chưa được verify.

------------------------------------------------------------------------

## 11. Điều kiện Duyên được phép bàn giao

### Exercise 1

Phải xác nhận:

-   Pretrained sentiment model load thành công.
-   Tokenizer load thành công.
-   Example sentences inference thành công.
-   Có tokens.
-   Có token IDs.
-   Có attention mask.
-   Có predicted label.
-   Có confidence score.
-   Phân biệt rõ prediction confidence và dataset accuracy.

### Exercise 2 --- Dataset

Phải xác nhận:

-   Rotten Tomatoes dataset load thành công.
-   Có train split.
-   Có validation split.
-   Có test split.
-   Labels đúng `0/1`.
-   Dataset schema đã được kiểm tra.
-   Missing/empty text đã được kiểm tra.
-   Label distribution đã được kiểm tra.
-   Duplicate analysis đã được thực hiện.
-   Token-length analysis đã được thực hiện.

### Preprocessing

Phải xác nhận:

-   DistilBERT tokenizer hoạt động.
-   `input_ids` tồn tại.
-   `attention_mask` tồn tại.
-   Labels hợp lệ.
-   Truncation configuration được xác định.
-   `max_length` có justification từ dữ liệu nếu sử dụng.
-   `DataCollatorWithPadding` hoạt động.
-   Decode sanity check pass.
-   Preprocessing sanity checks pass.

------------------------------------------------------------------------

## 12. Handoff --- Duyên → Viên

Sau khi hoàn thành Phase 6, Duyên phải tạo handoff document:

``` text
docs/result/phase_06_handoff_YYYY-MM-DD.md
```

Handoff document cần ghi:

### Completed Phases

``` text
Phase 0 → Phase 6
```

### Dataset

-   Dataset name
-   Train samples
-   Validation samples
-   Test samples
-   Labels

### Tokenizer

-   Tokenizer checkpoint
-   `max_length` / truncation configuration
-   Padding strategy

### Preprocessing Output

-   Available dataset objects
-   `input_ids`
-   `attention_mask`
-   labels
-   data collator

### Sanity Checks

Ghi rõ `PASS / FAIL` cho từng check.

### Important Decisions

Các quyết định từ Stage 1 có ảnh hưởng tới Phase 7 trở đi.

### Known Issues

Các lỗi hoặc vấn đề chưa giải quyết, nếu có.

### Files Created / Modified

Danh sách các file quan trọng.

### Last Commit

Commit hash dùng để bàn giao.

### Next Phase

``` text
Phase 7 — Model Construction
```

------------------------------------------------------------------------

## 13. Git trước khi Duyên bàn giao

Duyên kiểm tra:

``` bash
git status
```

Không để accidental changes chưa rõ nguồn gốc.

Chỉ stage các file liên quan:

``` bash
git add <ONLY_RELEVANT_FILES>
```

Commit:

``` bash
git commit -m "feat(practice3): complete phases 0-6"
```

Push:

``` bash
git push origin <working-branch>
```

Sau khi push, Duyên gửi cho Viên:

-   Branch name
-   Commit hash
-   Handoff document
-   Known issues, nếu có

------------------------------------------------------------------------

## 14. Viên nhận bàn giao

Viên không bắt đầu Phase 7 bằng cách sửa code ngay.

Thứ tự nhận bàn giao:

1.  Pull/fetch phiên bản mới nhất.
2.  Đọc handoff document.
3.  Đọc implementation Phase 6.
4.  Kiểm tra `git status`.
5.  Chạy lại notebook đến hết Phase 6.
6.  Xác nhận Phase 0--6 chạy được.
7.  Xác nhận preprocessing contract.
8.  Chỉ sau khi verification PASS mới bắt đầu Phase 7.

------------------------------------------------------------------------

## 15. Handoff Verification

Viên phải kiểm tra tối thiểu:

### Dataset

-   [ ] Train split tồn tại
-   [ ] Validation split tồn tại
-   [ ] Test split tồn tại

### Tokenizer

-   [ ] Đúng tokenizer/checkpoint

### Processed Sample

-   [ ] `input_ids`
-   [ ] `attention_mask`
-   [ ] label

### Batch

-   [ ] Dynamic padding hoạt động

### Labels

-   [ ] Labels thuộc `{0, 1}`

Nếu các điều kiện trên **FAIL**:

**Không bắt đầu Phase 7.**

Phải tạo error documentation và trao đổi lại với Duyên.

------------------------------------------------------------------------

## 16. Stage 2 --- Quy trình của Viên

Sau khi handoff verification PASS:

``` text
Phase 7 — Model Construction
        ↓
Phase 8 — Metrics & Training Configuration
        ↓
Phase 9 — Fine-Tuning
        ↓
Phase 10 — Learning Curves
        ↓
Phase 11 — Validation & Test Evaluation
        ↓
Phase 12 — Error Analysis
        ↓
Phase 13 — New-Sentence Inference
        ↓
Phase 14 — Save & Reload Model
```

Viên vẫn phải tuân thủ cho từng Phase:

``` text
PLAN → IMPLEMENT → NOTEBOOK CALL → VERIFY → RESULT → COMMIT
```

------------------------------------------------------------------------

## 17. Điều kiện hoàn thành Stage 2

Trước khi chuyển sang Phase 15 phải có:

-   Model construction sanity check PASS.
-   `logits.shape == [B, 2]`.
-   Loss finite.
-   Training completed.
-   Validation metrics.
-   Best checkpoint.
-   Independent test evaluation.
-   Accuracy.
-   Precision.
-   Recall.
-   F1-score.
-   Confusion Matrix.
-   Error Analysis.
-   Custom inference.
-   Saved model.
-   Saved tokenizer.
-   Reload verification.

------------------------------------------------------------------------

## 18. Final Stage --- Duyên + Viên

Sau Phase 14:

``` text
Duyên + Viên
      ↓
Review Phase 0–14
      ↓
Phase 15 — Final Summary
      ↓
Run All notebook
      ↓
Verify Results
      ↓
Write current_flow
      ↓
Final Audit
```

------------------------------------------------------------------------

## 19. Current Flow

Thư mục:

``` text
docs/current_flow/
```

chỉ được hoàn thiện **sau khi Practice 3 đã implementation xong**.

`current_flow` phải mô tả:

> What the final implementation actually does.

Không mô tả plan cũ nếu implementation cuối cùng đã thay đổi.

------------------------------------------------------------------------

## 20. Data Rule

Raw data:

``` text
data/raw_data/
```

Processed/split data:

``` text
data/data_after_split/
```

Không chỉnh sửa raw data trực tiếp nếu không có lý do và tài liệu ghi
nhận.

------------------------------------------------------------------------

## 21. Diagram Rule

Diagram sử dụng cho Practice 3 lưu tại:

``` text
image_diagram/
```

------------------------------------------------------------------------

## 22. Git Working Rule

Trước khi bắt đầu công việc:

``` bash
git pull
```

Chỉ commit file liên quan đến nhiệm vụ đang thực hiện.

Không dùng commit message chung chung như:

``` text
update
fix
done
```

Khuyến nghị:

``` text
feat(practice3): implement phase 05 dataset analysis
fix(practice3): fix tokenization preprocessing
docs(practice3): add phase 06 processing plan
```

Không sửa Phase của thành viên khác nếu chưa trao đổi.

Không force push vào branch làm việc chung.

------------------------------------------------------------------------

## 23. Definition of Done cho một Phase

Một Phase chỉ được xem là hoàn thành khi:

1.  Plan đã được lưu.
2.  Implementation đã chạy.
3.  Notebook gọi được module.
4.  Output đã được kiểm tra.
5.  Không còn runtime error chưa xử lý.
6.  Result/log cần thiết đã được lưu.

------------------------------------------------------------------------

## 24. Definition of Done cho Practice 3

Practice 3 chỉ hoàn thành khi:

-   Phase 0--15 hoàn thành.
-   Notebook chạy end-to-end.
-   Không còn unresolved runtime error.
-   Test set chỉ được sử dụng cho final evaluation.
-   Không fabricate metrics.
-   Results được lưu.
-   Model và tokenizer save thành công.
-   Reload verification thành công.
-   `current_flow` hoàn thành.
-   Duyên và Viên cùng review.
-   Final audit hoàn thành.

------------------------------------------------------------------------

## 25. Workflow Summary

``` text
                    DUYÊN
                      │
                 Phase 0 → 6
                      │
                Verify Phase 6
                      │
                 Commit + Push
                      │
              Handoff Document
                      │
                      ▼
                VIÊN VERIFY
                      │
               ┌──────┴──────┐
               │             │
             FAIL           PASS
               │             │
        Return / Fix      Phase 7 → 14
                             │
                             ▼
                      DUYÊN + VIÊN
                             │
                         Phase 15
                             │
                          Run All
                             │
                       current_flow
                             │
                         Final Audit
```
