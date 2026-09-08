# Phase 11 Read-Only NumPy Tensor Bridge Fix Plan

## Objective

Loại bỏ undefined-behavior warning tại NumPy to Torch boundary trong khi giữ feature timeline read-only.

## Impacted file

`src/course_work/data/datasets.py`

## Changes

Thay zero-copy conversion của `x` bằng tensor construction có bản sao float32 sở hữu độc lập.

## Impact

Mỗi sample tiếp tục được lazy slice, nhưng Tensor trả ra không thể mutate canonical timeline.

## Risk

Có thêm một lần copy trên từng sample. Đây là chi phí có chủ đích để bảo vệ tính bất biến và phù hợp với Dataset boundary.

## Validation

Chạy compile, synthetic Dataset/DataLoader smoke test với warning được nâng thành lỗi, kiểm tra shape, dtype, value equality và feature matrix bất biến.
