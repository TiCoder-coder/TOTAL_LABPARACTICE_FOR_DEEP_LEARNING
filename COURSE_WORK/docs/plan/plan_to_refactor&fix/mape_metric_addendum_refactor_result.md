# Kết quả refactor MAPE Metric Addendum

## 1. Phạm vi đã thực thi

Phần refactor bổ sung MAPE theo cơ chế hậu kiểm độc lập sau Phase 59. Công việc không tạo Phase 60, không thay đổi metric lựa chọn mô hình, không chạy training, không chạy lại Test inference, không load checkpoint và không fit lại scaler.

`RMSE Wh` tiếp tục là metric duy nhất dùng cho model selection, early stopping và BEST checkpoint. `MAPE (%)` chỉ là metric báo cáo bổ sung.

## 2. Kiến trúc sau refactor

### 2.1. Tầng evaluation

File `src/course_work/evaluation/metrics.py` chỉ sở hữu:

- Contract số học của MAPE.
- Công thức MAPE chuẩn trên dữ liệu Wh.
- Chuẩn hóa vector về NumPy float64.
- Kiểm tra finite, số lượng phần tử và target bằng 0.
- Kết quả có trạng thái xác định hoặc không xác định.

Chính sách target bằng 0 không dùng epsilon và không tự loại sample. Nếu quần thể có bất kỳ target bằng 0, MAPE của toàn quần thể có trạng thái `UNDEFINED_ZERO_TARGET`.

### 2.2. Tầng metric addendum

Thư mục `src/course_work/metric_addendum/` được tách theo trách nhiệm:

- `contract.py` quản lý contract, version và đường dẫn được phép.
- `sources.py` phát hiện prediction artifacts, đọc metric evidence và kiểm tra nguồn Test theo checksum registry.
- `compute.py` tính MAPE trên toàn bộ quần thể Validation đã căn chỉnh.
- `materialize.py` ghi artifact MAPE và processing log.
- `audit.py` kiểm toán quy trình machine learning độc lập với tính toán MAPE.

Tầng này chỉ đọc prediction artifacts đã tồn tại. Tầng này không import training, Phase 47 evaluator, checkpoint loader hoặc scaler fitting.

### 2.3. Tầng reporting

File `src/course_work/reporting/mape_addendum.py` chỉ đọc artifact và dựng HTML tĩnh. Renderer không tính lại metric, không ghi artifact, không dùng JavaScript và không dùng Jupyter widget.

### 2.4. Tầng notebook

`notebook_course_work/CourseWork_1.ipynb` được bổ sung một phần hậu kiểm cuối notebook. Cell code chỉ import và gọi `render_mape_addendum()`.

Cell không chứa công thức MAPE, source discovery, kiểm tra checksum, xử lý DataFrame, serialization, training hoặc inference.

## 3. Kết quả MAPE Validation

- Nguồn prediction Validation phát hiện: 19.
- Nguồn có MAPE xác định: 19.
- Nguồn có MAPE không xác định: 0.
- Nhóm mô hình có evidence: Persistence, LSTM và Transformer.
- Số Transformer run có prediction artifact: 17.
- Số sample của mỗi quần thể Validation: 2960.
- Giá trị target nhỏ nhất: 20 Wh.
- Số target bằng 0: 0.

Không tính mean và standard deviation trên 17 Transformer sweep run vì đây là các cấu hình khác nhau, không phải các lần lặp seed độc lập. Mean và sample standard deviation theo ba seed chỉ được phép tính nếu các prediction bundle Test tương ứng được khôi phục và vượt qua checksum gate.

## 4. Trạng thái MAPE Test

Checksum registry khai báo bốn prediction bundle Test:

- Persistence.
- Transformer seed 42.
- Transformer seed 123.
- Transformer seed 2026.

Bốn file prediction CSV không tồn tại ở đường dẫn canonical. Vì vậy:

- Trạng thái Test MAPE là `BLOCKED_SOURCE_UNAVAILABLE`.
- Không có Test MAPE được tính.
- Không chạy lại Test inference.
- Không load checkpoint.
- Không tái tạo prediction từ metric summary.

Các giá trị Test MAE, RMSE và R2 lịch sử vẫn giữ nguyên trong Phase 47–59 nhưng không đủ để suy ngược MAPE.

## 5. Kết quả kiểm toán quy trình

Audit gồm 21 hạng mục:

- `PASS`: 16.
- `PARTIAL`: 3.
- `BLOCKED_SOURCE_UNAVAILABLE`: 1.
- `NOT_APPLICABLE`: 1.
- `FAIL`: 0.

Ba hạng mục `PARTIAL`:

1. Canonical EDA là Train-only nhưng direct EDA cell trong notebook đọc full validated temporal view để hiển thị. Phần này chỉ được giữ vai trò mô tả và không được dùng cho feature selection, tuning hoặc threshold fitting.
2. Chính sách scaler là Train-only và Phase 9 signoff là PASS, nhưng checksum hiện tại của `artifacts/scaling/README_SCALING.md` khác checksum đã đóng băng trong Phase 9 signoff.
3. LSTM có so sánh trên Validation nhưng không được đánh giá trên final Test vì trạng thái eligibility là `NOT_EVALUATED`.

Hạng mục `BLOCKED_SOURCE_UNAVAILABLE` là MAPE Test do thiếu prediction CSV đã đóng băng.

Hạng mục `NOT_APPLICABLE` là inferential statistics hậu kiểm. Pipeline hiện báo cáo metric xác định và độ biến thiên mô tả ba seed; không tự ý bổ sung hypothesis test hoặc confidence interval sau khi Phase 59 đã đóng băng.

## 6. Artifact và log mới

Các artifact được lưu tại `artifacts/metric_addendum/mape/`:

- `mape_metric_contract.json`.
- `validation_mape_by_run.csv`.
- `validation_mape_summary.json`.
- `validation_source_audit.csv`.
- `final_test_source_audit.csv`.
- `final_test_mape_status.json`.
- `mape_discrepancies.json`.
- `mape_addendum_signoff.json`.
- `ml_pipeline_compliance_audit.csv`.
- `ml_pipeline_compliance_audit.json`.

Processing logs:

- `docs/save_log_in_processing/mape_metric_addendum_log.json`.
- `docs/save_log_in_processing/ml_pipeline_compliance_audit_log.json`.

## 7. Verification đã thực hiện

- Unit test công thức, zero-target policy, shape và deterministic contract.
- Integration test source-only materialization, idempotence và Test source blocking.
- Reporting test xác nhận HTML tĩnh, không widget và không script.
- Tổng số targeted test đạt: 10 trên 10.
- Compatibility suite kết hợp đạt 22 test và còn 1 test lỗi do checksum README Phase 9 đã tồn tại trước MAPE.
- Đối chiếu MAPE độc lập trên 19 file có sai số lớn nhất `3.552713678800501e-15`.
- Cell dashboard cuối notebook đã được chạy riêng với Jupyter kernel.
- 151 cell trước cell addendum được giữ nguyên.
- 73 output cũ trước addendum được giữ nguyên.
- Kernelspec `venv (3.10.11)` của người dùng được giữ nguyên.
- Checksum frozen Phase 12, Phase 47, Phase 58 và Phase 59 không thay đổi.

Lỗi compatibility còn lại được ghi tại `docs/plan-doc/analysis_error/phase_9_scaling_readme_checksum_mismatch_issue.md` và có corrective plan riêng tại `docs/plan-doc/plan_before_process/phase_9_scaling_readme_checksum_corrective_plan.md`. Không có file Phase 9 nào bị sửa trong lần refactor này.

## 8. Trạng thái kết thúc

MAPE Validation addendum hoàn thành. Pipeline compliance audit hoàn thành với trạng thái `PASS_WITH_LIMITATIONS`. Test MAPE chưa thể hoàn thành do thiếu frozen prediction bundles và không được phép khắc phục bằng cách chạy lại Test inference trong phạm vi này.
