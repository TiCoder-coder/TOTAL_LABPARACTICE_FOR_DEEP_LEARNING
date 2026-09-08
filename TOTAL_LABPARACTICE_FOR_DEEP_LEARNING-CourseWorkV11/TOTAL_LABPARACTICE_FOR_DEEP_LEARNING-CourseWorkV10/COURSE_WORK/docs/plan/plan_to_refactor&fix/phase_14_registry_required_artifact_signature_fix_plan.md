# Phase 14 Registry Required Artifact Signature Fix Plan

## Trạng thái

APPROVED_BY_ACTIVE_IMPLEMENTATION_FLOW

## Objective

Đồng bộ call site còn thiếu trong `validate_registry` với chữ ký `_required_artifacts(execution_type, model_family)` mà không thay đổi contract khác.

## File impact

Chỉ sửa:

- `src/course_work/experiments/registry.py`

Không sửa test để che failure.

## Thay đổi

Tại completed-record validation, truyền model family từ canonical run config cùng execution type vào `_required_artifacts`.

## Rủi ro

- Truy cập sai path model family có thể che malformed config.
- Persistence required predictions có thể không được validate nhất quán giữa `complete_run` và `validate_registry`.

## Kiểm soát

- Dùng cùng canonical path đã dùng trong `complete_run`.
- Chạy lại toàn bộ registry unit suite.
- Xác nhận test Persistence thiếu predictions vẫn fail completion.
- Xác nhận checksum mutation test quay lại hành vi mong đợi.

## Validation

1. Compile source và test module trong memory.
2. Chạy `tests/unit/test_experiment_registry.py`.
3. Chỉ đóng issue khi toàn bộ targeted tests pass.
