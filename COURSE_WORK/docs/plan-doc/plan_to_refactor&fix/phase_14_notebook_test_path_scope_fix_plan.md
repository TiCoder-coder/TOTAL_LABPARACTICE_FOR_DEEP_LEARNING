# Phase 14 Notebook Test Path Scope Fix Plan

## Trạng thái

APPROVED_BY_ACTIVE_IMPLEMENTATION_FLOW

## Objective

Sửa phạm vi path của lệnh inspection mà không sửa nội dung file.

## Thay đổi

- Giữ working directory là `COURSE_WORK`.
- Bỏ tiền tố `COURSE_WORK/` khỏi target path.
- Chạy lại lệnh read-only.

## Validation

- Lệnh kết thúc exit code 0.
- Nội dung cuối test file được đọc đầy đủ.

## Bổ sung sau lần tái diễn thứ 2

- Verify working directory bằng target path ngắn, không gắn project prefix.
- Chạy notebook inspection và tests trong cùng working directory với các path `notebook_course_work/...` và `tests/...`.
- Chỉ đóng lại issue sau khi cả jq và pytest đều exit code 0.
