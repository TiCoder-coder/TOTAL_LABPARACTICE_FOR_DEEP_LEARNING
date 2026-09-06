# Phase 12 Verification API Contract Name Fix Plan

## Objective

Kiểm tra sample-weighted epoch loss, Test firewall và baseline comparison bằng đúng public API Phase 12.

## Changes

Không sửa source hoặc artifact. Thay import sai bằng `aggregate_sample_weighted_loss`, dùng đúng các trường `split_id`, `evaluation_mode`, `run_id`, `model_id` của `EvaluationContext` và kiểm tra `delta_rmse_wh` theo comparison schema.

## Validation

Chạy lại toàn bộ smoke test Gate 2, đối chiếu metric reference, R² undefined state, residual convention, sample-weighted loss và Test firewall.
