# Phase 11 Artifact Inspection Interpreter Issue

## Hiện tượng

Một lệnh read-only inspection dùng system Python và không import được pandas.

## Nguyên nhân

Interpreter của lệnh không phải project virtual environment đã được Phase 1 khóa.

## Phạm vi ảnh hưởng

Phase 11 materialization, artifacts, idempotency và upstream checksum không bị ảnh hưởng.

## Yêu cầu sửa

Mọi Python verification command phải dùng `../venv/bin/python`.
