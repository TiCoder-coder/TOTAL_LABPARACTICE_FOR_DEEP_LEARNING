# Phase 11 Dataset Config Fingerprint Envelope Issue

## Hiện tượng

Real-data Dataset gate dừng tại `DatasetConfig` vì các lineage version fields được truyền như runtime constructor fields.

## Nguyên nhân

Một payload duy nhất đang được dùng đồng thời cho canonical fingerprint và dataclass construction dù hai schema có mục đích khác nhau.

## Phạm vi ảnh hưởng

Lỗi xảy ra trước khi Dataset suite được tạo và trước khi Phase 11 ghi artifact. Phase 9 và Phase 10 không bị thay đổi.

## Yêu cầu sửa

Version lineage phải tiếp tục tham gia dataset fingerprint. Runtime dataclass chỉ nhận đúng schema đã khai báo.
