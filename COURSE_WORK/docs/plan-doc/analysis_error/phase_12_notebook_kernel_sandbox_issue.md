# Phase 12 Notebook Kernel Sandbox Issue

## Hiện tượng

Notebook executor dừng trước cell đầu tiên với `PermissionError: Operation not permitted` khi Jupyter tìm cổng loopback cho kernel.

## Nguyên nhân

Filesystem sandbox cho phép đọc và ghi workspace nhưng không cho process bind socket local mà Jupyter kernel bắt buộc sử dụng.

## Phạm vi ảnh hưởng

Notebook source hợp lệ và notebook boundary tests đều PASS. Không cell nào được chạy trong lần thất bại nên không có mixed execution state hoặc artifact bị sửa một phần.

## Yêu cầu sửa

Chạy lại cùng lệnh notebook executor ngoài socket sandbox với project virtual environment, timeout hiện hành và không thay đổi notebook logic.
