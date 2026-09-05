# Phase 12 Verification API Contract Name Issue

## Hiện tượng

Lệnh smoke test import `compute_sample_weighted_epoch_loss` nhưng module Phase 12 cung cấp `aggregate_sample_weighted_loss`. Lệnh cũng khởi tạo `EvaluationContext` bằng `split` và `mode` thay vì `split_id` và `evaluation_mode`. Unit test comparison đọc `delta_rmse_wh_baseline_minus_model` thay vì field schema `delta_rmse_wh`.

## Nguyên nhân

Tên trong verification harness không khớp public API và comparison schema đã triển khai. Phase 12 chỉ quy định sample-weighted epoch loss theo công thức và không khóa tên helper này. Evaluation context dùng tên trường đầy đủ để tránh nhầm split và mode. Comparison schema quy định `delta_rmse_wh` là baseline trừ model.

## Phạm vi ảnh hưởng

Module Phase 12 import thành công. Logic metric, artifact và notebook chưa bị thay đổi bởi lỗi verification harness.

## Yêu cầu sửa

Verification harness phải import và gọi `aggregate_sample_weighted_loss` với batch means và batch sizes tách biệt, khởi tạo `EvaluationContext` bằng đúng schema và đọc field comparison chuẩn.
