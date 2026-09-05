# Phase Table Visible Text Hash Style Scope Fix Plan

## Trạng thái

APPROVED_BY_ACTIVE_IMPLEMENTATION_FLOW

## Objective

Thay phép đo lẫn CSS bằng content-preservation verification chỉ phản ánh text người dùng nhìn thấy.

## File impact

Không sửa production source hoặc test.

Chỉ cập nhật issue sau khi verification đạt.

## Thay đổi

Strip `style` block khỏi rendered HTML trước khi HTML parser thu thập text. Kết hợp kết quả với checksum processing logs, sign-off và unit allowlist tests.

## Rủi ro

- Loại bỏ nhầm visible content ngoài style.
- Chỉ kiểm tra tính lặp lại mà không kiểm tra canonical source.

## Kiểm soát

- Regex chỉ nhắm đúng `style` element.
- Kiểm tra repeated render cho toàn bộ Phase 0–14.
- So sánh processing-log và sign-off aggregate hashes với baseline đã chụp trước refactor.

## Validation

1. Hai lượt render sau khi strip style phải có cùng content hash.
2. Processing-log aggregate hash phải trùng baseline.
3. Sign-off aggregate hash phải trùng baseline.
4. Presentation unit suite phải tiếp tục pass.
