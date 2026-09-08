# Phase 14 Notebook JQ Root Selection Issue

## Trạng thái

RESOLVED

## Lỗi quan sát được

Lệnh inspection gọi `to_entries` trực tiếp trên notebook root object rồi giả định mỗi value là cell. Các entry như `metadata`, `nbformat` và `cells` không có cùng schema, dẫn tới lỗi:

`Cannot index array with string "source"`

## Root cause

Truy vấn thiếu bước chọn `.cells` trước khi duyệt `to_entries`.

## Ảnh hưởng

- Không đọc được cell index để chuẩn bị patch notebook.
- Notebook, source, artifacts và registry không bị thay đổi.

## Điều kiện đóng issue

- Truy vấn bắt đầu từ `.cells | to_entries`.
- Xác định chính xác import cell, Phase 13 cells và boundary marker.
- Chưa sửa notebook trước khi inspection pass.

## Kết quả xác minh

- Truy vấn đã bắt đầu từ `.cells | to_entries`.
- Import cell, Phase 13 cells và boundary marker được xác định chính xác.
- Inspection kết thúc không có jq error.
