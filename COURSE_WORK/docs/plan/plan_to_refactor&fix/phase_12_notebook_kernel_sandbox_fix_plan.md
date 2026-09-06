# Phase 12 Notebook Kernel Sandbox Fix Plan

## Objective

Thực thi toàn bộ CourseWork notebook bằng kernel thật mà không thay đổi source flow.

## Changes

Không sửa source, artifact hoặc notebook cell. Chỉ cấp quyền cho Jupyter bind local kernel socket trong lần execute.

## Validation

Xác minh mọi code cell có execution count tuần tự, không có error output, không có warning stream, Phase 12 render METRICS-v1 PASS và notebook boundary tests tiếp tục PASS.
