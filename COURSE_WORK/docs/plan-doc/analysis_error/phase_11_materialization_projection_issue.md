# Phase 11 Materialization Projection Issue

## Hiện tượng

Phase 11 audit dừng trước artifact write vì Dataset equivalence probe dùng read-only NumPy bridge và loader registry projection thiếu trường trạng thái.

## Nguyên nhân

Probe chưa áp dụng ownership rule đã khóa cho Dataset output. LoaderConfig mô tả runtime configuration, còn `status` là field của artifact registry nên không tồn tại trong dataclass.

## Phạm vi ảnh hưởng

Không có Phase 11 artifact nào được ghi. Checksum tổng hợp của toàn bộ upstream artifacts vẫn giữ nguyên.

## Yêu cầu sửa

Probe phải tạo owned float32 Tensor. Registry projection phải bổ sung trạng thái audit tại artifact boundary mà không làm nhiễu runtime dataclass.
