# Phase 11 Dataset Config Fingerprint Envelope Fix Plan

## Objective

Tách canonical fingerprint envelope khỏi runtime DatasetConfig schema.

## Impacted file

`src/course_work/data/datasets.py`

## Changes

Tạo runtime payload chỉ gồm dataclass fields và tạo fingerprint từ runtime payload cộng các upstream version bindings.

## Impact

DatasetConfig khởi tạo hợp lệ trong khi fingerprint vẫn truy nguyên WINDOWS-v1, WINDOWPOP-v1, FEATURESETS-v1, SCALING-v1 và SPLIT-v1.

## Validation

Chạy compile và real-data Dataset gate cho ba split, xác minh sample counts, item keys, shapes, dtypes và baseline DataLoader batch.
