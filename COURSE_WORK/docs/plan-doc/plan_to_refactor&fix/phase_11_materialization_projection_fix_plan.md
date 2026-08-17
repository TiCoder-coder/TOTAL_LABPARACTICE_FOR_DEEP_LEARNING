# Phase 11 Materialization Projection Fix Plan

## Objective

Sửa ownership trong equivalence probe và hoàn thiện loader registry schema.

## Impacted file

`src/course_work/data/datasets.py`

## Changes

Tensor hóa direct materializer probe bằng bản sao float32 và gán `status=PASS` khi project LoaderConfig sang registry row.

## Impact

Runtime LoaderConfig vẫn chỉ chứa configuration semantics. Artifact registry có đủ schema theo Phase 11 contract.

## Validation

Xác nhận không có partial artifact, kiểm tra upstream checksum, chạy compile, unit tests và materialize toàn bộ Phase 11 từ đầu.
