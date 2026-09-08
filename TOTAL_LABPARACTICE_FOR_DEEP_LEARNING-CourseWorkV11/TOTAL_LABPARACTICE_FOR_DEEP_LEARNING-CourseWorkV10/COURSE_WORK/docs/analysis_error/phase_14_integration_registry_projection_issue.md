# Phase 14 Integration Registry Projection Issue

## Trạng thái

RESOLVED

## Bối cảnh

Final integration chain kiểm tra Phase 14 qua signed baseline artifacts và production experiment registry sau khi canonical Persistence run hoàn tất.

## Lỗi quan sát được

Integration suite có 12 test pass và 1 test fail.

Failure xuất hiện khi test đọc `test_access_authorized` trực tiếp từ một row của `experiment_registry.csv`:

`KeyError: 'test_access_authorized'`

## Root cause

Canonical JSONL run record chứa `test_access_authorized`, nhưng flattened registry CSV chỉ là inspection projection theo `MAIN_REGISTRY_COLUMNS` và không công bố field này. Test đã yêu cầu một field ngoài schema của projection.

## Phạm vi ảnh hưởng

- Một assertion trong integration chain không đọc đúng source of truth.
- Final regression bị chặn tại bước kiểm tra Test firewall.

## Phạm vi không ảnh hưởng

- Canonical Persistence run record.
- Phase 14 sign-off.
- Validation predictions và metrics.
- Test firewall runtime.
- Raw data và signed artifacts Phase 10–13.

## Điều kiện đóng issue

- Kiểm tra Test authorization từ Phase 14 sign-off hoặc canonical JSONL record.
- Flattened CSV chỉ được kiểm tra các field thuộc schema công bố.
- Chạy lại integration chain và toàn bộ test phải pass.

## Kết quả xác minh

- Assertion ngoài flattened CSV schema đã được loại bỏ.
- Test firewall vẫn được xác minh qua signed Phase 14 field `test_access_authorized = false` và `test_targets_materialized = false`.
- Integration chain đạt 13/13 test.
