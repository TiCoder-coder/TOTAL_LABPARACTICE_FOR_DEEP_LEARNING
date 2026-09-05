# Phase 13 Synthetic Run Root Path Fix Plan

## Objective

Cho phép synthetic registry sử dụng run directory tạm biệt lập mà không nới path ra ngoài phạm vi đã khai báo.

## Changes

Thêm explicit `run_root` vào canonical base-path resolution của artifact writer và artifact validator.

## Validation

Chạy lại reference config validation, toàn bộ synthetic lifecycle audit, path checksum audit và xác minh không có production run hoặc artifact nào được tạo.
