# Phase 14 Registry Required Artifact Signature Issue

## Trạng thái

RESOLVED

## Bối cảnh

Bước 1 mở rộng `ExperimentRegistry._required_artifacts` để required artifacts có thể phụ thuộc vào `model_family`. Persistence evaluation phải có cả `METRICS` và `PREDICTIONS`, trong khi các model family khác giữ contract hiện tại.

## Lỗi quan sát được

Targeted registry suite có 18 test pass và 1 test fail.

Failure xuất hiện trong `ExperimentRegistry.validate_registry` khi validate một completed SANITY run:

`TypeError: ExperimentRegistry._required_artifacts() missing 1 required positional argument: 'model_family'`

## Root cause

`complete_run` đã được cập nhật sang chữ ký mới và truyền `record["config"]["model"]["model_family"]`. Nhánh completed-record validation trong `validate_registry` vẫn gọi `_required_artifacts` chỉ với `execution_type`.

## Phạm vi ảnh hưởng

- Registry validation không thể hoàn tất khi có completed record.
- Artifact checksum regression test bị chặn trước khi kiểm tra checksum.
- Persistence implementation chưa bắt đầu và production registry chưa bị thay đổi.

## Phạm vi không ảnh hưởng

- Run registration.
- Status transitions.
- Metric registration.
- Persistence config validation.
- Raw data và artifacts Phase 10–13.

## Điều kiện đóng issue

- Mọi call site của `_required_artifacts` truyền cả `execution_type` và `model_family`.
- Targeted registry suite pass toàn bộ.
- Persistence completion vẫn bắt buộc predictions và metrics.
- SANITY/TRAINING/FINAL_TEST contracts không đổi.

## Kết quả xác minh

- Mọi call site đã dùng cùng canonical signature.
- Registry unit suite đạt 19/19 test.
- Persistence evaluation thiếu predictions vẫn bị chặn.
- Completed-run checksum validation hoạt động trở lại.
