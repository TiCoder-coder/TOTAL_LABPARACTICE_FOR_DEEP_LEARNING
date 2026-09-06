# Phase Table Visible Text Hash Style Scope Issue

## Trạng thái

RESOLVED

## Bối cảnh

Table-layout verification cần chứng minh nội dung hiển thị của Phase 0–14 không đổi sau CSS refactor.

## Lỗi quan sát được

Mọi table schema và layout class đều đúng nhưng visible-text hash sau refactor không trùng baseline.

## Root cause

HTML parser đưa nội dung của thẻ `style` vào text sequence. CSS là phần bắt buộc thay đổi của refactor nên hash đã đo cả presentation implementation thay vì chỉ đo visible phase content.

## Phạm vi ảnh hưởng

- Phép so sánh visible-text hash hiện tại không hợp lệ.
- Bước content-preservation verification chưa thể đóng.

## Phạm vi không ảnh hưởng

- Rendered phase values.
- Processing logs.
- Canonical artifacts và sign-off.
- Compact/wide table classification.
- Unit presentation suite.

## Điều kiện đóng issue

- Loại bỏ toàn bộ `style` content trước khi thu thập visible text.
- Chứng minh visible text ổn định qua repeated render.
- Chứng minh processing logs và sign-off giữ nguyên checksum baseline.
- Chứng minh `_phase_content`, allowlist và escaping tests tiếp tục pass.

## Kết quả xác minh

- Hai lượt render sau khi loại bỏ `style` cùng tạo content hash `798eef9cd8a7282caf3ad1a4f0bc21f1ac86f3a3c1214f940cac6f127e23da26`.
- Processing-log aggregate hash trùng baseline.
- Sign-off aggregate hash trùng baseline.
- Presentation unit suite đạt 16/16 test.
