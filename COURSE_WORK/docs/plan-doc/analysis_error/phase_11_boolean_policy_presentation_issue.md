# Phase 11 Boolean Policy Presentation Issue

## Hiện tượng

Minimal HTML table hiển thị `False` thành `FAIL`, làm `shuffle=False` và `drop_last=False` trông như audit failure dù đó là policy hợp lệ.

## Nguyên nhân

Generic boolean renderer dùng PASS/FAIL cho validation fields nhưng Phase 11 Loader policy đang đưa configuration booleans trực tiếp vào renderer.

## Phạm vi ảnh hưởng

Chỉ visible presentation bị diễn giải sai. Manifest, loader behavior, audits và sign-off đều đúng.

## Yêu cầu sửa

Configuration booleans trong Loader policy phải được project thành nhãn Yes hoặc No trước khi render.
