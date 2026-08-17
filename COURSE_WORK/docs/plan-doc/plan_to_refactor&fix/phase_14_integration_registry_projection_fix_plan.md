# Phase 14 Integration Registry Projection Fix Plan

## Trạng thái

APPROVED_BY_ACTIVE_IMPLEMENTATION_FLOW

## Objective

Đồng bộ integration assertion với ownership của registry source mà không mở rộng flattened CSV schema và không giảm mức bảo vệ Test firewall.

## File impact

Chỉ sửa:

- `tests/integration/test_phase_0_to_5_chain.py`

## Thay đổi

Giữ các assertion về run identity, family, execution type và completed status trên `experiment_registry.csv`. Kiểm tra Test authorization bằng signed Phase 14 fields đã được checksum bảo vệ.

## Rủi ro

- Xóa nhầm toàn bộ Test firewall assertion.
- Làm test phụ thuộc vào implementation detail không thuộc public artifact contract.

## Kiểm soát

- Giữ assertion `test_access_authorized = false` trên Phase 14 sign-off.
- Giữ assertion `test_targets_materialized = false` trên sign-off và manifest.
- Không sửa production code hoặc artifact schema.

## Validation

1. Chạy lại riêng integration chain.
2. Xác nhận 13/13 test pass.
3. Tiếp tục final regression chỉ sau khi targeted suite đạt.
