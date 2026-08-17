# Phase 11 Notebook Kernel Sandbox Fix Plan

## Objective

Hoàn tất end-to-end notebook execution bằng project kernel.

## Changes

Không sửa source hoặc notebook cells. Chỉ chạy lại nbconvert execute với quyền kernel socket phù hợp.

## Validation

Xác minh mọi code cell có execution count tuần tự, không có error output hoặc stream traceback, và Phase 11 chỉ sinh một HTML summary theo presentation allowlist.
