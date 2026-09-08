# Pre-process Plan: Phase 9 Scaling README Checksum Corrective

## 1. Trạng thái

Plan này được tạo từ lỗi compatibility test phát hiện trong quá trình bổ sung MAPE. Plan chưa được phép thực thi vì liên quan tới artifact đã ký của Phase 9 và cần Human phê duyệt riêng.

## 2. Vấn đề cần sửa

`artifacts/scaling/phase_9_signoff.json` và `docs/save_log_in_processing/phase_9_train_only_scaling_log.json` cùng ghi checksum mong đợi của `artifacts/scaling/README_SCALING.md` là:

`52edfe46135533583eae041a77bd5b21bbf016e71e7ce8b85b08ec9e9754970c`

Checksum hiện tại của file là:

`79ddc8f5a357e5a9ae233c0e52823d82c565a7a2e872c714c17dacbcb2c0dab7`

Các scaler, registry, statistics và leakage audit không được kết luận sai chỉ từ mismatch của README. Tuy nhiên loader strict của Phase 9 từ chối toàn bộ bundle khi bất kỳ output checksum nào lệch.

## 3. Mục tiêu corrective

- Xác định nguồn gốc chính xác của thay đổi README.
- Phân biệt thay đổi chỉ mang tính tài liệu với thay đổi scientific contract.
- Khôi phục tính nhất quán mà không sửa mù quáng signoff.
- Không fit lại scaler nếu scaler scientific payload vẫn hợp lệ.
- Không làm thay đổi downstream model, prediction hoặc Test evidence.

## 4. Trình tự đề xuất

### Bước 1. Preservation gate

- Chụp checksum toàn bộ Phase 9 artifacts.
- Chụp checksum Phase 10–59 signoff liên quan.
- Không ghi file trong bước kiểm tra.

### Bước 2. README provenance audit

- Đọc code sinh `README_SCALING.md` trong `src/course_work/data/scaling.py`.
- So sánh nội dung hiện tại với Git history, log Phase 9 và mọi bản lưu trữ có thể tồn tại.
- Xác định README hiện tại được sinh bởi code canonical hay được chỉnh ngoài flow.

### Bước 3. Scientific payload verification

- Kiểm tra checksum của scaler joblib, registry, scaling manifest và statistics theo Phase 9 signoff.
- Chạy kiểm tra train-only fit, feature order, round-trip và finite values mà không refit.
- Dừng nếu bất kỳ scientific artifact nào lệch.

### Bước 4. Chọn corrective mode

Nếu tìm được README canonical có checksum mong đợi và nội dung phù hợp, khôi phục đúng byte của README đó.

Nếu README hiện tại là phiên bản canonical mới nhưng chỉ thay đổi tài liệu, không sửa signoff cũ tại chỗ. Tạo amendment có version mới, ghi provenance, checksum mới và compatibility decision theo architecture rule sau khi được Human duyệt.

Nếu thay đổi ảnh hưởng scientific policy, mở corrective phase đầy đủ; không tiếp tục bằng document-only amendment.

### Bước 5. Verification

- Chạy lại `test_scaling.py`.
- Chạy lại `test_metrics.py`.
- Xác nhận `load_validated_target_scaler` hoạt động.
- Xác nhận tất cả Phase 10–59 frozen checksum không đổi.
- Chạy riêng cell Phase 9 nếu notebook cần cập nhật presentation; không chạy lại toàn notebook.

### Bước 6. Documentation

- Ghi corrective result tại `docs/plan-doc/plan_to_refactor&fix/`.
- Cập nhật issue với nguyên nhân gốc, mode đã chọn và verification evidence.

## 5. Điều cấm

- Không thay checksum trong signoff chỉ để test chuyển sang PASS.
- Không overwrite README trước khi xác định provenance.
- Không fit lại scaler khi chưa chứng minh scientific payload sai.
- Không chạy lại training.
- Không chạy lại Test inference.
- Không sửa Phase 47–59 artifacts hoặc conclusions.

## 6. Điều kiện cần Human duyệt

Human cần duyệt corrective mode sau Bước 2 và Bước 3 trước khi bất kỳ file Phase 9 nào được sửa.
