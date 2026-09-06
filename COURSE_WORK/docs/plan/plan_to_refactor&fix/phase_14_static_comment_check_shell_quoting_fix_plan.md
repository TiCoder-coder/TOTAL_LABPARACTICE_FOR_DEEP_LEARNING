# Phase 14 Static Comment Check Shell Quoting Fix Plan

## Trạng thái

APPROVED_BY_ACTIVE_IMPLEMENTATION_FLOW

## Objective

Thay lệnh regex không shell-safe bằng hai kiểm tra độc lập và xác định chính xác comment token.

## File impact

Không sửa source hoặc test.

Chỉ bổ sung issue và fix plan của verification flow.

## Thay đổi

Chạy `rg` với pattern không chứa nested quote, sau đó chạy tokenizer trên danh sách file Phase 14 bị tác động.

## Rủi ro

- `rg` đơn lẻ có thể bỏ sót inline comment.
- Tìm ký tự `#` thô có thể nhầm nội dung string với comment.

## Kiểm soát

- Tokenizer là source of truth cho Python comment token.
- Kiểm tra icon được chạy riêng theo Unicode range.

## Validation

1. `rg` comment-line check chạy không lỗi.
2. Tokenizer trả về zero comment token.
3. Unicode scan trả về zero icon.
