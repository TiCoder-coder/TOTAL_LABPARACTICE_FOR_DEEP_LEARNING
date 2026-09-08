# Phase 11 Notebook Kernel Sandbox Issue

## Hiện tượng

Jupyter notebook execution không thể bind local kernel ports và dừng với `Operation not permitted`.

## Nguyên nhân

Filesystem sandbox cho phép đọc và ghi workspace nhưng chặn socket bind cần thiết cho Jupyter kernel process.

## Phạm vi ảnh hưởng

Notebook vẫn ở clean state. Source, artifacts và tests không bị thay đổi.

## Yêu cầu sửa

Chạy đúng notebook execution command ngoài network sandbox với quyền chỉ đủ để Jupyter bind loopback ports.
