# MAPE Metric Addendum Issue

## Trạng thái hiện tại

`METRICS-v1` chỉ định MAE, RMSE và R² là ba metric bắt buộc. Validation RMSE là metric duy nhất dùng cho model selection, early stopping và checkpoint selection.

Phase 12 đã mô tả MAPE nhưng khóa chính sách ở `DISABLED_SUPPLEMENTARY_ONLY`. Phase 47 chỉ cho phép MAE, RMSE và R² tại lần mở Test chính thức. Phase 58 và Phase 59 đã hoàn tất, đồng thời scientific narrative đã được khóa.

## Vấn đề cần xử lý

Yêu cầu mới cần bổ sung MAPE để báo cáo sai số tương đối. Nếu sửa trực tiếp artifacts Phase 47–59 hoặc coi MAPE là metric đã đăng ký trước Test thì sẽ làm sai lịch sử thực nghiệm và vi phạm frozen contract.

Các frozen Test prediction paths được khai báo trong `artifacts/final_test/prediction_checksums.json` nhưng các CSV tương ứng không tồn tại trong workspace. Không được chạy lại Test inference để tái tạo chúng trong thay đổi này.

## Quyết định phạm vi

MAPE được bổ sung dưới dạng supplementary metric addendum. MAPE không được dùng để chọn model, thay đổi winner, thay đổi final lock, chọn seed, tạo ensemble hoặc sửa kết luận khoa học đã khóa.

`METRICS-v1`, `FINAL_TEST_EVAL-v1`, `FINAL_TABLES-v2` và `FINAL_CONCLUSIONS-v2` được giữ nguyên. Implementation mới sử dụng `METRICS-v2` cho những metric result mới và `POSTHOC-MAPE-v1` cho evidence được tính từ prediction artifacts đã tồn tại và qua kiểm tra checksum.

## Chính sách mẫu số

MAPE dùng công thức chuẩn trên original Wh. Không thêm epsilon và không loại bỏ target bằng zero. Nếu có bất kỳ target bằng zero, kết quả MAPE của population được đánh dấu `UNDEFINED_ZERO_TARGET`.

Raw dataset hiện có minimum Appliances bằng 10 Wh và zero-target count bằng 0. Chính sách vẫn phải được kiểm tra trên từng prediction population trước khi tính.

## Tác động kiến trúc

Thay đổi cần tách riêng các trách nhiệm:

- `evaluation` định nghĩa và kiểm tra MAPE.
- `metric_addendum` đọc frozen prediction artifacts, xác minh provenance và tạo derived artifacts.
- `reporting` chỉ đọc addendum artifacts và tạo HTML.
- Notebook chỉ gọi public renderer.
- Process audit chỉ đọc contracts và artifacts để đánh giá compliance.

## Điều kiện dừng

Nếu prediction source thiếu, checksum sai, population sai, target unit không phải Wh hoặc xuất hiện Test inference path thì scope tương ứng phải dừng với trạng thái `BLOCKED_SOURCE_UNAVAILABLE`. Không được thay thế bằng số liệu gõ tay, số liệu đọc từ hình hoặc inference mới.
