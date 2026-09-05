# Phase 11 Artifact Inspection Interpreter Fix Plan

## Objective

Chạy artifact inspection bằng đúng Phase 1 interpreter.

## Changes

Không sửa source hoặc artifact. Chỉ thay executable của verification command bằng project venv Python.

## Validation

Đọc manifest và toàn bộ audit CSV, xác minh row counts và mọi status đều PASS.
